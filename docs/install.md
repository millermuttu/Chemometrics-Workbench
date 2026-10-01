# Install

Download the package for your system from the
[latest release](https://github.com/millermuttu/Chemometrics-Workbench/releases):

| System | Package |
| --- | --- |
| Windows 10 or 11, 64-bit | `ChemometricsWorkbench-windows-x64.zip` |
| macOS on Apple Silicon | `ChemometricsWorkbench-macos-arm64.dmg` |
| Linux, x86-64 | `ChemometricsWorkbench-linux-x64.tar.gz` |

You do not need Python, Node or anything else installed. The package carries all of it.

The packages are **not code-signed**, so Windows and macOS will warn you the first time you open
one. **[How to open this](how-to-open.md)** walks through that warning on each system and gives the
download and unpacked sizes.

## What happens when it starts

1. A console window opens and prints a line like
   `Launch URL: http://127.0.0.1:52817/?token=…`.
2. Your default browser opens at that address, already signed in.
3. The workbench opens its default project and creates it the first time. The project lives in
   `.config/chemometrics-workbench/projects/default` under your home folder on every system. On
   Linux, `$XDG_CONFIG_HOME` moves it if you have set that variable.

**To stop the workbench, close the console window.** Closing the browser tab does not stop it.

The port changes on every start, and the token is new each time. If you close the tab, copy the
`Launch URL` from the console again, because an address without its token is refused.

## From source

Developers can run it from a checkout with `./run.sh`. See the
[README](https://github.com/millermuttu/Chemometrics-Workbench#running-it).
