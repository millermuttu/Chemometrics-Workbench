# The HTML report

Each run can be saved as **one standalone HTML file**. The report records what the run did, what
data it ran against, how it scored, and its scores and results plots. The file carries everything
inside itself, with no server, no network and no scripts from elsewhere. You can email it, archive
it, or open it on a machine that has never seen the workbench, and it still shows its plots.

!!! note "No report button yet"
    The report is reachable over the application's HTTP interface only for now. A button for it is
    tracked in [#247](https://github.com/millermuttu/Chemometrics-Workbench/issues/247).

## Saving one

With the workbench running, take the port and token from the console's `Launch URL`
(`http://127.0.0.1:<port>/?token=<token>`) and run, in a terminal:

```bash
curl -H "Authorization: Bearer <token>" -OJ \
  http://127.0.0.1:<port>/api/experiments/current/report.html
```

`current` is the latest run. Any experiment's id works in its place, and
`GET /api/experiments` lists them. `-OJ` saves the file under the name the workbench gives it.
Open the file in any browser.

If a run's dataset has since been removed from the project, its report is refused with a sentence
saying so. The report has to name what the run was measured against, and without the dataset it
cannot.
