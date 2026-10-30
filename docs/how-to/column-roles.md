# Change a column's role

When a file is imported, each column that is not a spectrum is given a role by what it holds: a column of numbers is a **target**, which a regression models; a column of text is **metadata**, and metadata with two or more values is offered as a **class**, which a classifier separates. The guess is usually right, and sometimes not:

- classes coded as numbers - variety 1, 2, 3 - are read as a target, and no classifier is offered;
- an identifier or a batch is read as a class, and offered to a classifier first if it comes first;
- a file with no target offers no regression at all.

None of these needs the file edited and imported again.

## In Dataset view

Open the dataset. Each column's header has a **role** menu: target, class, group or ignore.

| Role | What it is for |
| --- | --- |
| target | a reference value a regression models |
| class | a column a classifier separates, two or more values |
| group | a column a split keeps together - replicates, batches - and not offered as a class |
| ignore | left out of the version |

Choosing a role writes a **new version** of the dataset, as excluding samples does: the spectra are the parent's, the column has moved, and the change is recorded beside the version ("Changed from v1: variety → class"). The pipeline moves onto the new version, and the step menu follows it: a model the dataset has no column for is listed disabled, with the reason and a link back here.

A text column cannot become a target; the refusal names the first sample that is not a number. A class needs two values. To undo a change, put the pipeline back on the parent version from its Dataset view.

The rules, and what each version records, are in [`design/data-model.md`](https://github.com/millermuttu/Chemometrics-Workbench/blob/dev/design/data-model.md); the endpoint is `POST /api/datasets/{id}/roles`.
