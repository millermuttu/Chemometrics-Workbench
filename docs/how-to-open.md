# How to open Chemometrics Workbench

The packages on the [releases page](https://github.com/millermuttu/Chemometrics-Workbench/releases)
are **not code-signed**. Signing needs a paid Apple developer account and a Windows certificate,
and 1.0 ships without them (`PROPOSAL.md` §4.2 and §17). The application is the same either way.
Your system will warn you the first time you open it, and this page shows you how to get past that
warning.

**Size.** The download is 45–70 MB depending on the platform. Unpacked, it takes about 230 MB on
Linux. Each release's notes list the measured download and unpacked size for every platform.

When it starts, a console window opens and your default browser opens on the workbench. **Close the
console window to stop the workbench.** If the browser does not open, copy the `Launch URL` the
console prints into your browser. The URL includes the token, so the address alone will not work.

## Windows

1. Download `ChemometricsWorkbench-windows-x64.zip`, right-click it and choose **Extract All**.
2. Open the extracted `ChemometricsWorkbench` folder and double-click `ChemometricsWorkbench.exe`.
3. SmartScreen says **"Windows protected your PC"**. Click **More info**, then **Run anyway**.
   Windows remembers this choice, so you only do it once.

## macOS (Apple Silicon)

1. Download `ChemometricsWorkbench-macos-arm64.dmg` and open it.
2. Drag the `ChemometricsWorkbench` folder to **Applications** (or anywhere you like).
3. In that folder, double-click `ChemometricsWorkbench`. Gatekeeper says the app **"cannot be
   opened"** or that Apple **"could not verify"** it. Click **Done**, not **Move to Trash**.
4. Open **System Settings → Privacy & Security**, scroll to the message about
   `ChemometricsWorkbench`, and click **Open Anyway**. Confirm with your password.
   On macOS 14 and earlier, **right-click → Open** in Finder does the same.

The application opens in Terminal, and Terminal is its console window.

If macOS keeps refusing, clear the download quarantine from the whole folder once:

```bash
xattr -dr com.apple.quarantine /Applications/ChemometricsWorkbench
```

Intel Macs are not supported by the 1.0 packages.

## Linux

```bash
tar -xzf ChemometricsWorkbench-linux-x64.tar.gz
./ChemometricsWorkbench/ChemometricsWorkbench
```

The terminal you start it from is its console window.
