# The HTML report

Each run can be saved as **one standalone HTML file**. The report records what the run did, what
data it ran against, how it scored, and its scores and results plots. The file carries everything
inside itself, with no server, no network and no scripts from elsewhere. You can email it, archive
it, or open it on a machine that has never seen the workbench, and it still shows its plots.

## Saving one

Open a run from **Experiments** in the project outline and click **Save report**. Your browser
saves `experiment-<date>-<time>-<id>.html`. Open it in any browser.

If a run's dataset has since been removed from the project, the report is refused, and the sentence
saying why appears beside the button. The report has to name what the run was measured against,
and without the dataset it cannot.
