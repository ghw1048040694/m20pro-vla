# M20 Pro Isaac Sim asset bridge

The canonical robot description remains `src/m20pro_description/urdf/M20.urdf`.
This bridge stages that URDF and its 17 STL meshes on the Windows D: drive,
then imports them using the installed Isaac Sim 6.x URDF importer. It does not
modify the Isaac Sim installation or claim that the imported physics has been
validated for training.

From WSL2, in the `m20pro-ros2-sim` repository:

```bash
python scripts/isaac/m20pro_asset.py stage \
  --source src/m20pro_description \
  --destination /mnt/d/M20Pro-RL/asset
```

From a **Windows Command Prompt** after Isaac Sim starts successfully:

```bat
D:\isaacsim\python.bat D:\M20Pro-RL\asset\m20pro_asset.py import --asset-dir D:\M20Pro-RL\asset --usd-dir D:\M20Pro-RL\asset\usd
```

The import writes `usd_import_report.json` only after it opens the USD,
finds all 16 named joints, and finds an articulation root. The URDF's own
primitive collision geometries are retained; visual STL meshes are not
converted into collision meshes. The base remains floating and self-collision
is disabled for the first locomotion smoke test. The four wheel joints remain
continuous, not locked as leg joints.

Before RL training, validate wheel contact direction, joint target mode,
standing pose, mass/COM, wheel friction, actuator limits, and a short passive
drop/commanded-roll test in the imported asset. An import report alone does
not establish dynamic equivalence with MuJoCo or real hardware.

The local Windows install at `D:\isaacsim` was checked on 2026-09-28. It reports
`6.0.1-rc.7` but cannot start its base Kit experience because its required
`isaacsim.anim.robot.schema` extension is missing. The import command currently
stops before reading the URDF; use a complete matching Isaac Sim installation
before retrying. Do not patch the vendor install to hide that dependency.
