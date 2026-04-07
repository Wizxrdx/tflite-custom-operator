# TFLite GridSample Runtime Build (Standalone custom_ops Folder)

This folder builds a TensorFlow Lite C runtime DLL where `TFLiteGridSample` is
compiled in and registered by default in the resolver. There is no separate
custom-op plugin DLL in this flow.

## What This Folder Does

- Fetches TensorFlow Lite source with CMake `FetchContent`.
- Compiles `grid_sample.cc` into the `tensorflow-lite` runtime target.
- Patches TFLite `BuiltinOpResolver` at configure time to add:
  - `AddCustom("TFLiteGridSample", Register_GRID_SAMPLE())`
- Builds:
  - `custom_ops/build/Release/tensorflowlite_c.dll`
- Validates end-to-end with `custom_ops/examples/test_dll.ipynb`.

## What This Folder Does Not Do

- No standalone plugin build (`grid_sample_op.dll`).
- No runtime `AddCustomOp` registration from an external DLL.

## Step-by-Step: Register a Custom Op (GridSample Example)

This repository uses **compile-time registration**: the custom op is built into
the TFLite runtime and added to the default resolver.

1. **Define an exported registration function**

  In `custom_ops/grid_sample.h`, expose the op registration entrypoint:

  ```cpp
  GRID_SAMPLE_EXPORT TfLiteRegistration* Register_GRID_SAMPLE();
  ```

2. **Implement the kernel and registration object**

  In `custom_ops/grid_sample.cc`:

  - Implement `Init`, `Free`, `Prepare`, and `Invoke`.
  - Return a static `TfLiteRegistration` from `Register_GRID_SAMPLE()`.
  - Set `custom_name` to the exact op name in the `.tflite` model.

  ```cpp
  extern "C" {
  GRID_SAMPLE_EXPORT TfLiteRegistration* Register_GRID_SAMPLE() {
    static TfLiteRegistration reg = {
      tflite::ops::custom::grid_sample::Init,
      tflite::ops::custom::grid_sample::Free,
      tflite::ops::custom::grid_sample::Prepare,
      tflite::ops::custom::grid_sample::Invoke,
    };
    reg.custom_name = "TFLiteGridSample";
    reg.version = 1;
    return &reg;
  }
  }
  ```

3. **Compile the custom op into TensorFlow Lite**

  In `custom_ops/CMakeLists.txt`, add the op sources to `tensorflow-lite`:

  ```cmake
  target_sources(tensorflow-lite PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/grid_sample.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/grid_sample.h
  )
  target_include_directories(tensorflow-lite PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  ```

4. **Register the op in the default resolver**

  This flow patches upstream `tensorflow/lite/core/kernels/register.cc` during
  configure to inject:

  ```cpp
  TfLiteRegistration* Register_GRID_SAMPLE();
  AddCustom("TFLiteGridSample", tflite::ops::custom::Register_GRID_SAMPLE());
  ```

  After this, interpreters that use `BuiltinOpResolver` can resolve
  `TFLiteGridSample` without loading a separate plugin DLL.

5. **Build the runtime DLL**

  From repo root:

  ```powershell
  Remove-Item custom_ops/build -Recurse -Force -ErrorAction Ignore
  cmake -S custom_ops -B custom_ops/build -G "Visual Studio 17 2022" -A x64
  cmake --build custom_ops/build --config Release
  ```

6. **Validate with the sample notebook**

  Run `custom_ops/examples/test_dll.ipynb` (Cells 1-6). Success indicates the
  runtime resolved `TFLiteGridSample` via default resolver registration.

### Important Naming Rule

- The string in registration must match exactly across all points:
  - Model custom op code: `"TFLiteGridSample"`
  - Resolver registration: `AddCustom("TFLiteGridSample", ...)`
  - Registration object: `reg.custom_name = "TFLiteGridSample"`

If these differ, TFLite reports unresolved custom op errors at interpreter
creation time.

## Build Requirements (Windows)

- Visual Studio 2022 Build Tools (Desktop C++ workload)
- CMake 3.16+
- Git
- Python 3.10+

## Build (From Repo Root)

```powershell
Remove-Item custom_ops/build -Recurse -Force -ErrorAction Ignore
cmake -S custom_ops -B custom_ops/build -G "Visual Studio 17 2022" -A x64
cmake --build custom_ops/build --config Release
```

Expected artifact:

- `custom_ops/build/Release/tensorflowlite_c.dll`

## Optional Export Sanity Check

You can quickly confirm C API exports are present:

```powershell
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$dumpbin = Get-ChildItem "$install\VC\Tools\MSVC" -Recurse -Filter dumpbin.exe | Select-Object -First 1 -ExpandProperty FullName
& $dumpbin /exports "custom_ops/build/Release/tensorflowlite_c.dll" | Select-String "TfLiteVersion|TfLiteModelCreateFromFile|TfLiteInterpreterCreate|Register_GRID_SAMPLE"
```

## Notebook Validation

1. Open `custom_ops/examples/test_dll.ipynb`.
2. Run Cell 1 through Cell 6 in order.
3. Confirm the last cell prints:
   - `SUCCESS: tensorflowlite_c.dll executed TFLiteGridSample via default resolver registration.`

This confirms the custom op is resolved by the runtime default resolver (no
external op DLL load path).

## Troubleshooting

- `LNK1104 cannot open ...\tensorflowlite_c.dll` during build:
  - The DLL is usually in use (Python or notebook process still holds it).
  - Close/restart the notebook kernel or stop Python processes using the DLL, then rebuild.
- `AttributeError: function 'TfLiteVersion' not found` in notebook:
  - You are likely loading an old/incorrect DLL. Rebuild and ensure the path points to `custom_ops/build/Release/tensorflowlite_c.dll`.
- Notebook Cell 6 shows `NameError` for setup variables:
  - Run earlier cells first; Cell 6 depends on setup done in Cells 1-5.
- Stock `tf.lite.Interpreter` fails with unresolved custom op:
  - This is expected for this test model and proves the custom op is not available in the stock interpreter runtime.

## Notes

- This flow does not depend on the top-level `tensorflow/` folder in the repo.
- Keep `GIT_TAG` pinned in `custom_ops/CMakeLists.txt` for reproducible builds.
