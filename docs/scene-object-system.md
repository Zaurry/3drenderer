# Scene object system

The viewer now separates the editable scene from the renderer's flat runtime
scene:

```text
OBJ files/directories
        |
        v
SceneDocument
  - deduplicated mesh assets
  - object hierarchy
  - local TRS transforms
  - visibility and locking
  - per-object material overrides
  - point/directional lights
  - environment
  - undo/redo history
        |
        v
flattened Scene render snapshot
        |
        +-- CPU raster
        +-- CPU ray/path
        +-- CUDA path
        `-- OpenGL/GLSL
```

Keeping a renderer-facing snapshot makes object edits behave identically in
all four viewer modes. An OBJ is loaded only once into the document asset
cache; duplicating an object creates another object that references the same
asset. Geometry is transformed and flattened only when the document changes.
This is the compatibility layer for the existing render backends; true
GPU/CPU draw instancing and a two-level render BVH can be added later without
changing the document or `.rscene` format.

## Opening assets

`--asset` is repeatable and accepts either an OBJ file or a directory:

```powershell
.\build\default\bin\viewer.exe `
  --asset "Computer Graphics Archive\hw1\model-a.obj" `
  --asset "Computer Graphics Archive\hw1\model-b.obj" `
  --mode raster
```

To recursively import every OBJ below a directory:

```powershell
.\build\default\bin\viewer.exe `
  --asset "Computer Graphics Archive\hw1" `
  --mode raster
```

Directory structure becomes object hierarchy. Files are sorted by relative
path before import, making object creation deterministic.

The viewer also supports:

- **Import OBJ...** for one or more files.
- **Import folder...** for recursive import.
- Dragging OBJ files, directories, or `.rscene` files onto the window.
- `--scene-file path\to\scene.rscene` for opening a saved scene directly.

## Editing

The **Scene Objects** panel contains the Outliner and Inspector.

- Click selects one object; `Ctrl+click` toggles multi-selection.
- Drag an Outliner row onto another row to reparent it while preserving its
  world transform.
- `G`, `R`, and `S` choose translate, rotate, and scale gizmos.
- The Inspector edits exact local translation, Euler rotation, and scale.
- The Inspector's **Materials** section selects OBJ/MTL material slots by
  material name. Editing creates an override for the active object only.
- `Ctrl+D` duplicates the selected subtree. Mesh duplicates share the same
  loaded asset and initially copy the same overrides; subsequent edits remain
  independent.
- `Delete` removes the selected subtree.
- `F` frames the active object.
- `Ctrl+Z` and `Ctrl+Y` undo and redo scene edits.
- Visibility is inherited: hiding a group hides every descendant.
- Locked objects cannot edit or reset material overrides.

### Material overrides

An override can change the material type, base color/tint, roughness, IOR,
emission, opacity, alpha cutoff, bump scale, and two-sided state. Diffuse,
opacity, and bump texture switches are shown when the source material provides
those maps and start enabled. The base color/tint is multiplied by the
original diffuse texture, so changing a color does not discard `map_Kd`.

**Reset override** removes the active object's override and restores the
currently loaded OBJ/MTL values. The original OBJ, MTL, and texture files are
never modified. The first editor version intentionally does not replace
texture files.

Raster and OpenGL update base color, textures, emission, opacity, bump, and
two-sided preview state immediately. Metal roughness and dielectric IOR retain
their physical meaning in Ray and Path modes; this does not turn the raster
backends into a full PBR pipeline.

Selection outlines and the transform gizmo are editor overlays. They do not
modify the linear framebuffer or path-tracing accumulation samples.

## Scene files

`.rscene` version 2 is a versioned JSON scene document. It stores:

- asset paths, relative to the scene file when possible;
- stable object and parent IDs;
- object names, types, local transforms, visibility, and locking;
- non-empty per-object material overrides;
- light color/intensity and the environment.

Version 1 scene files remain readable and open with no material overrides.
Overrides whose material slot is no longer present are retained in the file,
ignored while rendering, and reported as warnings.

Use **Open scene...**, **Save**, and **Save as...** in the panel. `Ctrl+O` and
`Ctrl+S` are available while the panel is open. Saving writes a temporary file
first and then replaces the destination. Missing assets are reported as
warnings and their object entries remain in the Outliner, so the hierarchy is
not silently discarded.

The scene file references OBJ/MTL/texture assets; it does not copy them.
Keeping the `.rscene` near its asset directory makes the relative references
portable.

## Current render boundary

The authoring side is asset/object based, while render backends still consume
`Scene::triangles`. Rebuilding a snapshot is proportional to the total number
of visible triangle instances. This is suitable for editing the homework
scenes and guarantees feature parity today. Large scenes with many repeated
instances will benefit from a later BLAS/TLAS runtime representation.
