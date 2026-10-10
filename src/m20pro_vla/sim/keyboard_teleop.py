"""Human keyboard demonstrations, with synchronized pre-action onboard sensors.

One physics owner; HTTP handlers only update a latest-input mailbox. Demonstrations
remain outside episode_*.npz training discovery until reviewed and contracted.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import queue
import secrets
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from m20pro_vla.low_level.factory import build_low_level_controller
from m20pro_vla.low_level.shield import lidar_safety_shield
from m20pro_vla.sim.mujoco import (
    ASSET, ObjectSpec, ObstacleSpec, SceneLightSpec, build_scene,
    planar_lidar, proprioception,
)

STOP = np.array([0., 0., 0., 1.])
KEYS = {'KeyW', 'KeyS', 'KeyA', 'KeyD', 'ShiftLeft', 'ShiftRight', 'Space'}
CONTRACT = 'm20_human_keyboard_physical_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + '\n', encoding='utf-8')
    temporary.replace(path)


def keyboard_request(keys, fresh):
    """Body units: m/s, m/s, rad/s, stop. Release/timeout means stop."""
    keys = set(keys) & KEYS
    if not fresh or 'Space' in keys:
        return STOP.copy()
    forward = int('KeyW' in keys) - int('KeyS' in keys)
    turn = int('KeyA' in keys) - int('KeyD' in keys)
    if not forward and not turn:
        return STOP.copy()
    fast = bool(keys & {'ShiftLeft', 'ShiftRight'})
    vx = forward * ((.5 if fast else .35) if forward > 0 else (.25 if fast else .20))
    yaw = turn * (.15 if forward else .4)
    if forward and turn:
        vx = math.copysign(min(abs(vx), .18), vx)
    return np.array([vx, 0., yaw, 0.])


def executed_request(request, previous, scan):
    if request[3] >= .5:
        return STOP.copy(), 'none'
    target = request.copy()
    # Avoid fast drive plus large turn; decelerate before an in-place turn and
    # unwind an existing large turn before accelerating. These are v8 limits,
    # independent of task identity or visual target detection.
    if abs(previous[0]) > .06 and abs(target[2]) > .15:
        target[2] = math.copysign(.15, target[2])
    if abs(previous[2]) > .15 and abs(target[0]) > .18:
        target[0] = math.copysign(.18, target[0])
    moving = target.copy()
    moving[:3] = previous[:3] + np.clip(target[:3] - previous[:3], [-.01, 0., -.0075], [.01, 0., .0075])
    return lidar_safety_shield(moving, scan, stop_distance=.5, slow_distance=1.25)


class ChunkWriter:
    """Bounded raw RGB writer; chunks survive interrupted or aborted recordings."""
    def __init__(self, folder):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.q = queue.Queue(maxsize=2)
        self.error = None
        self.index = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while True:
                job = self.q.get()
                try:
                    if job is None:
                        return
                    index, arrays = job
                    path = self.folder / f'chunk_{index:05d}.npz'
                    with path.with_suffix('.tmp').open('wb') as f:
                        np.savez_compressed(f, **arrays)
                    path.with_suffix('.tmp').replace(path)
                finally:
                    self.q.task_done()
        except Exception as exc:
            self.error = exc

    def append(self, row):
        if self.error:
            raise self.error
        self.rows.append(row)
        if len(self.rows) == 250:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        arrays = {k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]}
        while True:
            if self.error:
                raise self.error
            try:
                self.q.put((self.index, arrays), timeout=.1)
                break
            except queue.Full:
                pass
        self.index += 1
        self.rows = []

    def finish(self):
        self.flush()
        while self.q.unfinished_tasks:
            if self.error:
                raise self.error
            time.sleep(.02)
        self.q.put(None)
        self.thread.join(timeout=10)
        if self.error:
            raise self.error
        return [{'path': str(p), 'sha256': sha(p)} for p in sorted(self.folder.glob('chunk_*.npz'))]


class Recorder:
    def __init__(self, config, run_path):
        self.config, self.run_path = config, Path(run_path)
        self.settings = config['teleoperation']
        self.output = Path(self.settings['output_dir'])
        self.output.mkdir(parents=True, exist_ok=True)
        source = Path(self.settings['source_dir']).resolve()
        protected = Path(config['paths']['dataset']).resolve()
        if source != protected or source == self.output.resolve():
            raise ValueError('Manual demonstrations must use the protected training scene source and a separate output')
        self.episodes = []
        for ident in self.settings['source_episode_ids']:
            p = source / f'episode_{int(ident)}.json'
            metadata = json.loads(p.read_text())
            if not 6012 <= int(metadata['layout_id']) <= 6023 or int(ident) in range(21006, 21024):
                raise ValueError('Only reviewed training layouts are enabled for demonstrations')
            self.episodes.append((p, metadata, sha(p)))
        self.token = secrets.token_urlsafe(24)
        self.lock = threading.Lock()
        self.keys, self.input_at, self.owner = [], 0., None
        self.operations = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.jpeg = {}
        self.status = {'mode': 'loading', 'error': None}
        self.writer = None
        self.task_index = 0
        self.previous = STOP.copy()
        self.samples = 0
        self.saved = []
        self.session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + secrets.token_hex(3)
        self.model = self.data = self.renderer = self.controller = None
        self.base_receipt = {
            'schema': CONTRACT, 'training_ready': False,
            'policy_onnx_sha256': sha(config['low_level']['policy_onnx']),
            'asset_sha256': sha(ASSET), 'config': config,
            'server_pid': __import__('os').getpid(), 'session': self.session,
            'operator_observation': ['front_rgb', 'rear_rgb', 'task_text'],
            'policy_arrays': ['front_rgb', 'rear_rgb', 'lidar', 'proprio', 'action'],
            'diagnostic_only': ['requested_action', 'pre_xy', 'post_xy', 'qpos', 'qvel', 'ctrl', 'contact', 'shield', 'input_age_s', 'wall_time_ns'],
            'source_fps': 50, 'frame_stride_for_future_training': 2,
            'action_units': ['m/s', 'm/s', 'rad/s', 'stop'],
            'action_limits': {'forward': [-.25, .5], 'yaw': [-.4, .4]},
            'watchdog_s': .3, 'v8_retrained': False,
            'render_profile': 'interactive-v1: egl-llvmpipe, no shadows/reflections/MSAA',
        }
        dump(self.output / f'session_{self.session}.json', self.base_receipt)

    def publish(self, **values):
        with self.lock:
            self.status.update(values)

    def input(self, body):
        now = time.monotonic()
        with self.lock:
            client = str(body.get('client', ''))[:100]
            if not client:
                raise ValueError('Missing client identity')
            if self.owner and self.owner != client and now - self.input_at < .5:
                raise ValueError('Another tab currently holds keyboard control')
            self.owner, self.input_at = client, now
            self.keys = list(set(body.get('keys', [])) & KEYS)

    def command(self, body):
        operation = body.get('operation')
        if operation not in {'start', 'save', 'retry', 'next', 'shutdown'}:
            raise ValueError('Unknown operation')
        with self.lock:
            self.keys, self.input_at = [], 0.
        self.operations.put_nowait(operation)

    def load(self):
        self.publish(mode='loading')
        if self.renderer:
            self.renderer.close()
        p, metadata, source_sha = self.episodes[self.task_index]
        self.metadata = metadata
        self.source_path, self.source_sha = p, source_sha
        self.scene = self.output / f'scene_{self.session}_{self.task_index}.xml'
        objects = [ObjectSpec(**{**o, 'kind': 'cylinder' if 'cylinder' in o['label'] else 'box'}) for o in metadata['objects']]
        obstacles = [ObstacleSpec(**o) for o in metadata['obstacles']]
        build_scene(self.scene, objects, obstacles=obstacles, task_object_collisions=False,
                    light=SceneLightSpec(**metadata['scene_light']))
        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.data = mujoco.MjData(self.model)
        # Preserve the pinned scene/asset friction exactly; no gait adaptation.
        self.model.vis.quality.offsamples = 0
        self.controller = build_low_level_controller(self.model, teacher_motion_limits=(.5, .4), inference_threads=1)
        self.controller.reset(self.data, float(metadata['initial_yaw']), np.asarray(metadata['initial_xy']))
        for _ in range(int(metadata.get('warmup_steps', 35))):
            self.controller.step(self.data, np.zeros(4))
        self.renderer = mujoco.Renderer(self.model, height=96, width=160)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        from OpenGL import GL
        self.base_receipt['gl_renderer'] = GL.glGetString(GL.GL_RENDERER).decode()
        self.obstacle_ids = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, o.name + '_geom') for o in obstacles}
        self.previous = STOP.copy()
        self.samples = 0
        self.capture()
        self.publish(mode='ready', source_episode_id=metadata['episode_id'], task=metadata['task_text'],
                     task_number=self.task_index + 1, task_count=len(self.episodes), steps=0,
                     sim_seconds=0, measured_fps=0, action=STOP.tolist(), shield='none',
                     saved=self.saved, error=None, recording_id=None)

    def capture(self):
        images = {}
        for camera, name in [('front_rgb', 'front'), ('rear_rgb', 'rear')]:
            self.renderer.update_scene(self.data, camera=camera)
            rgb = self.renderer.render().copy()
            images[name] = rgb
            b = io.BytesIO()
            Image.fromarray(rgb).save(b, format='JPEG', quality=85)
            with self.lock:
                self.jpeg[name] = b.getvalue()
        return images

    def begin(self):
        if self.writer:
            return
        self.record_id = f'human_{self.session}_{len(self.saved):03d}'
        self.folder = self.output / self.record_id
        self.writer = ChunkWriter(self.folder)
        self.samples, self.contacts, self.stop_tail = 0, 0, 0
        self.min_height, self.max_roll, self.max_pitch = float('inf'), 0., 0.
        self.rms = np.zeros(3)
        self.min_distance = float('inf')
        self.started_wall, self.started_sim = time.monotonic(), self.data.time
        # Source privileged labels stay only in metadata; no routes/GT are used
        # by keyboard execution or displayed to the operator.
        self.record_meta = {**self.base_receipt, 'recording_id': self.record_id,
            'source_episode_id': self.metadata['episode_id'], 'source_metadata': self.metadata,
            'source_json_sha256': self.source_sha, 'scene_sha256': sha(self.scene),
            'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'recording'}
        dump(self.folder / 'metadata.json', self.record_meta)
        self.publish(mode='recording', recording_id=self.record_id)

    def finish(self, disposition):
        if not self.writer:
            return
        self.publish(mode='saving')
        chunks = self.writer.finish()
        rms = np.sqrt(self.rms / max(1, self.samples))
        final_distance = float(np.linalg.norm(self.data.qpos[:2] - np.asarray(self.metadata['target_xy_privileged_label_only'])))
        checks = {'zero_obstacle_contact': self.contacts == 0,
            'height': self.min_height >= .45, 'roll': self.max_roll <= 8,
            'pitch': self.max_pitch <= 8, 'vertical_rms': rms[0] <= .12,
            'roll_rate_rms': rms[1] <= .28, 'pitch_rate_rms': rms[2] <= .25,
            'terminal_stop20': self.stop_tail >= 20, 'arrival': final_distance <= .95,
            'operator_completed': disposition == 'operator_complete', 'nonempty': self.samples > 0}
        result = {**self.record_meta, 'status': disposition, 'steps': self.samples,
            'finished_at': datetime.now(timezone.utc).isoformat(), 'training_ready': False,
            'sim_seconds': float(self.data.time - self.started_sim),
            'wall_seconds': time.monotonic() - self.started_wall, 'chunks': chunks,
            'actual_contact_steps': self.contacts, 'min_height': self.min_height if self.samples else None,
            'max_abs_roll_deg': self.max_roll, 'max_abs_pitch_deg': self.max_pitch,
            'rms_vertical_roll_pitch': rms.tolist(), 'min_target_distance': self.min_distance if self.samples else None,
            'final_target_distance': final_distance, 'terminal_stop_steps': self.stop_tail,
            'review_checks': checks, 'physical_task_review_passed': all(checks.values())}
        dump(self.folder / 'metadata.json', result)
        self.saved.append({'id': self.record_id, 'disposition': disposition,
                           'steps': self.samples, 'review_passed': all(checks.values())})
        self.writer = None
        self.previous = STOP.copy()
        self.publish(mode='saved', saved=self.saved, action=STOP.tolist(), last_result=self.saved[-1])

    def step(self):
        tick_started = time.monotonic()
        with self.lock:
            keys, input_age = self.keys[:], tick_started - self.input_at
        requested = keyboard_request(keys, input_age <= .3)
        scan = planar_lidar(self.model, self.data)
        action, shield = executed_request(requested, self.previous, scan)
        scan_finished = time.monotonic()
        images = self.capture()
        render_finished = time.monotonic()
        row = {'front_rgb': images['front'], 'rear_rgb': images['rear'],
            'lidar': np.asarray(scan, dtype=np.float32),
            'proprio': np.asarray(proprioception(self.model, self.data), dtype=np.float32),
            'action': action.astype(np.float32), 'requested_action': requested.astype(np.float32),
            'step': np.int64(self.samples), 'pre_xy': self.data.qpos[:2].copy(),
            'qpos': self.data.qpos.copy(), 'qvel': self.data.qvel.copy(),
            'sim_time': np.float64(self.data.time), 'wall_time_ns': np.int64(time.time_ns()),
            'input_age_s': np.float32(min(input_age, 1e6)),
            'input_timeout': np.bool_(input_age > .3), 'shield': np.asarray(shield)}
        diagnostics = self.controller.step(self.data, action)
        physics_finished = time.monotonic()
        contacts = any(int(self.data.contact[i].geom1) in self.obstacle_ids or
                       int(self.data.contact[i].geom2) in self.obstacle_ids for i in range(self.data.ncon))
        roll, pitch = self.controller._attitude(self.data)
        row.update(post_xy=self.data.qpos[:2].copy(), contact=np.bool_(contacts), ctrl=self.data.ctrl.copy(),
                   low_level_recovery=np.bool_(diagnostics.safety_recovery_active))
        self.writer.append(row)
        self.previous = action
        self.samples += 1
        self.contacts += int(contacts)
        self.stop_tail = self.stop_tail + 1 if action[3] >= .5 and np.all(action[:3] == 0) else 0
        self.min_height = min(self.min_height, float(self.data.qpos[2]))
        self.max_roll = max(self.max_roll, abs(math.degrees(roll)))
        self.max_pitch = max(self.max_pitch, abs(math.degrees(pitch)))
        self.rms += np.square(self.data.qvel[[2, 3, 4]])
        self.min_distance = min(self.min_distance, float(np.linalg.norm(self.data.qpos[:2] - np.asarray(self.metadata['target_xy_privileged_label_only']))))
        elapsed = max(.001, time.monotonic() - self.started_wall)
        self.publish(steps=self.samples, sim_seconds=round(self.samples * .02, 2),
                     measured_fps=round(self.samples / elapsed, 1), action=action.tolist(),
                     shield=shield, input_timeout=input_age > .3, contact_steps=self.contacts,
                     speed_mps=round(float(np.linalg.norm(row['post_xy'] - row['pre_xy']) / .02), 3))
        self.publish(timing_ms={'lidar': round(1000 * (scan_finished-tick_started), 2),
                     'render': round(1000 * (render_finished-scan_finished), 2),
                     'physics': round(1000 * (physics_finished-render_finished), 2)})
        if self.samples >= int(self.settings.get('max_steps', 19000)):
            self.finish('budget_reached')
        time.sleep(max(0., .02 - (time.monotonic() - tick_started)))

    def run(self):
        try:
            self.load()
            last_receipt = 0.
            while not self.closed.is_set():
                try:
                    op = self.operations.get_nowait()
                except queue.Empty:
                    op = None
                if op == 'start':
                    if not self.writer:
                        if self.status['mode'] == 'saved':
                            self.load()
                        self.begin()
                elif op == 'save':
                    self.finish('operator_complete')
                elif op in {'retry', 'next'}:
                    self.finish('operator_retry' if op == 'retry' else 'operator_next')
                    if op == 'next':
                        self.task_index = (self.task_index + 1) % len(self.episodes)
                    self.load()
                elif op == 'shutdown':
                    self.finish('session_closed')
                    self.closed.set()
                if self.writer:
                    self.step()
                else:
                    time.sleep(.02)
                if time.monotonic() - last_receipt > 1:
                    with self.lock:
                        status = dict(self.status)
                    dump(self.run_path / 'teleop_status.json', {**status, 'pid': __import__('os').getpid(),
                         'checked_at': datetime.now(timezone.utc).isoformat(), 'output': str(self.output)})
                    last_receipt = time.monotonic()
        except BaseException as exc:
            self.publish(mode='error', error=f'{type(exc).__name__}: {exc}')
            if self.writer:
                self.finish('interrupted')
            raise
        finally:
            if self.renderer:
                self.renderer.close()


def run_keyboard_recorder(config, run_path):
    recorder = Recorder(config, run_path)
    page = Path(__file__).with_name('keyboard_teleop.html').read_text(encoding='utf-8').replace('__TOKEN__', recorder.token)
    port = int(config['teleoperation'].get('port', 8765))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, code, content, mime='application/json'):
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            route = self.path.split('?')[0]
            if route == '/':
                return self.respond(200, page.encode(), 'text/html; charset=utf-8')
            if route == '/api/status':
                with recorder.lock:
                    payload = dict(recorder.status)
                return self.respond(200, json.dumps(payload, ensure_ascii=False).encode())
            if route in {'/camera/front.jpg', '/camera/rear.jpg'}:
                with recorder.lock:
                    frame = recorder.jpeg.get(route.split('/')[-1][:-4])
                return self.respond(200 if frame else 503, frame or b'', 'image/jpeg')
            self.respond(404, b'{}')

        def do_POST(self):
            try:
                if self.headers.get('Host') not in {f'localhost:{port}', f'127.0.0.1:{port}'}:
                    raise ValueError('Localhost only')
                origin = self.headers.get('Origin')
                if origin and origin not in {f'http://localhost:{port}', f'http://127.0.0.1:{port}'}:
                    raise ValueError('Cross-origin control is disabled')
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4096:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(size))
                if not secrets.compare_digest(str(body.get('token', '')), recorder.token):
                    raise ValueError('Invalid control token')
                if self.path == '/api/input':
                    recorder.input(body)
                elif self.path == '/api/operation':
                    recorder.command(body)
                else:
                    raise ValueError('Unknown endpoint')
                self.respond(200, b'{"ok":true}')
            except (ValueError, TypeError, queue.Full) as exc:
                self.respond(400, json.dumps({'error': str(exc)}).encode())

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    dump(Path(run_path) / 'teleop_endpoint.json', {'url': f'http://localhost:{port}', 'pid': __import__('os').getpid(),
         'output': str(recorder.output), 'session': recorder.session})
    try:
        recorder.run()
    finally:
        server.shutdown()
        server.server_close()
    return {'schema': CONTRACT, 'recordings': recorder.saved, 'training_ready': False, 'output': str(recorder.output)}
