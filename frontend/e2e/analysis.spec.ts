import { readFile } from "node:fs/promises";

import { expect, test, type Page } from "@playwright/test";

/** The analysis tab: the artboard's panel grid, the T² ellipse drawn from the
 * served limit, and an outlier that can be hovered to name its sample. */

async function openResults(page: Page) {
  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /PCA 5 PC/ }).first().dblclick();
  await expect(page.getByTestId("scores-plot")).toBeVisible();
  await expect(page.locator(".gl-container canvas").first()).toBeVisible();
}

test("one tab holds the panel grid that answers question 11.1", async ({ page }) => {
  await openResults(page);
  for (const panel of ["Scores", "Loadings", "Explained variance", "Diagnostics"]) {
    await expect(page.getByText(panel, { exact: true })).toBeVisible();
  }
  // The headline numbers the artboard puts in the header, in the accent ink.
  const header = page.getByTestId("analysis-header");
  await expect(header).toContainText("PCA 5 components · 240 × 100");
  await expect(header).toContainText("PC1");
  await expect(header).toContainText("68.9%");
  await expect(header).toContainText("99.9%");
});

test("the axes carry their components and the loadings axis its real unit", async ({ page }) => {
  await openResults(page);
  await expect(page.getByTestId("scores-plot")).toContainText("PC 1 (68.9%)");
  await expect(page.getByTestId("scores-plot")).toContainText("PC 2 (28.4%)");
  await expect(page.getByTestId("loadings-plot")).toContainText("wavelength_nm (nm)");

  await page.getByLabel("Scores y axis").selectOption("2");
  await expect(page.getByTestId("scores-plot")).toContainText("PC 3 (1.6%)");
});

test("the ellipse comes from the fixture's limit, not from the browser", async ({ page }) => {
  await openResults(page);
  const measured = await page.evaluate(() => {
    const plot = document.querySelector("[data-testid=scores-plot]") as HTMLElement & {
      data?: { name?: string; x?: number[]; y?: number[] }[];
    };
    const ellipse = (plot.data ?? []).find((trace) => trace.name === "T² limit");
    return { x: Math.max(...(ellipse?.x ?? [])), y: Math.max(...(ellipse?.y ?? [])) };
  });

  const served = await page.evaluate(async () => {
    const response = await fetch("/api/results/pca_a", {
      headers: { Authorization: `Bearer ${sessionStorage.getItem("token")}` },
    });
    const pca = await response.json();
    return {
      x: Math.sqrt(pca.diagnostics.hotelling_t2_limit * pca.eigenvalues[0]),
      y: Math.sqrt(pca.diagnostics.hotelling_t2_limit * pca.eigenvalues[1]),
    };
  });

  expect(measured.x).toBeCloseTo(served.x, 9);
  expect(measured.y).toBeCloseTo(served.y, 9);
});

test("the diagnostics block lists what the limits put outside, in tabular numerals", async ({
  page,
}) => {
  await openResults(page);
  await expect(page.getByText("Hotelling T² limit")).toBeVisible();
  await expect(page.getByText("SPE limit")).toBeVisible();

  const rows = page.locator("table tbody tr");
  await expect(rows.first()).toBeVisible();
  const alignment = await page
    .locator("table td.n")
    .first()
    .evaluate((cell) => getComputedStyle(cell).fontVariantNumeric);
  expect(alignment).toContain("tabular-nums");
});

test("hovering a score names its sample", async ({ page }) => {
  await openResults(page);
  const plot = page.getByTestId("scores-plot");
  const box = (await plot.boundingBox())!;
  // Sweep the middle of the cloud until Plotly puts a hover label up.
  for (let fraction = 0.3; fraction < 0.7; fraction += 0.02) {
    await page.mouse.move(box.x + box.width * fraction, box.y + box.height * 0.5);
    if (await page.locator(".hovertext").count()) break;
  }
  await expect(page.locator(".hovertext")).toBeVisible();
  await expect(page.locator(".hovertext")).toContainText(/C\d{3}|E\d{3}/);
});

test("a regression tab draws what a decomposition has no counterpart for", async ({ page }) => {
  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /PLS 5 LV/ }).first().dblclick();

  // The shared half renders, because a regression and a decomposition mean the
  // same thing by scores, loadings and both diagnostics.
  await expect(page.getByTestId("scores-plot")).toBeVisible();
  for (const panel of ["Scores", "Loadings", "Explained variance", "Diagnostics"]) {
    await expect(page.getByRole("region", { name: panel })).toBeVisible();
  }

  // And the half it does not: the row Phase 1.1 reserved in a comment. By
  // role and name rather than by text - "RMSECV" is also an axis title, a
  // metric row and a header figure on this same screen.
  for (const panel of ["Predicted vs measured", "RMSECV", "Calibration metrics"]) {
    await expect(page.getByRole("region", { name: panel })).toBeVisible();
  }
  await expect(page.getByTestId("predicted-plot")).toBeVisible();
  await expect(page.getByTestId("rmsecv-plot")).toBeVisible();

  // #184: VIP beside the loadings, on the same axis - one point per variable.
  await expect(page.getByRole("region", { name: "Variable importance" })).toBeVisible();
  await expect(page.getByTestId("vip-plot")).toBeVisible();
  const points = await page.evaluate(() => {
    const plot = document.querySelector("[data-testid=vip-plot]") as HTMLElement & {
      data?: { x?: number[] }[];
    };
    return plot.data?.[0]?.x?.length ?? 0;
  });
  expect(points).toBe(100);

  // The seeded chain has an SNV in it, which is not a fixed linear map, so
  // the raw-axis coefficients are refused - and the refusal names the step.
  await page.getByLabel("Variable importance view").selectOption("coefficients");
  const refused = page.getByTestId("coefficients-unavailable");
  await expect(refused).toBeVisible();
  await expect(refused).toContainText("SNVTransformer cannot be folded");

  // The header says what the model is and leads with the two numbers that say
  // whether it generalises, rather than PC1 and cumulative variance.
  const header = page.getByTestId("analysis-header");
  await expect(header).toContainText("PLS on fat 5 components · 216 × 100");
  await expect(header).toContainText("RMSECV");
  await expect(header).toContainText("Q²");

  // A metric the payload carries is a number; one it omits is an em dash, per
  // metrics-and-validation.md section 11. This node is below a split, so
  // RMSECV is present - and nothing here is ever rendered as 0.0000.
  await expect(page.getByTestId("metric-RMSECV")).not.toHaveText("—");
  await expect(page.getByTestId("metric-Q²")).not.toHaveText("—");
});

test("a decomposition tab is unchanged, and shows none of the regression panels", async ({
  page,
}) => {
  await openResults(page);
  for (const panel of [
    "Predicted vs measured",
    "RMSECV",
    "Calibration metrics",
    "Variable importance",
  ]) {
    await expect(page.getByRole("region", { name: panel })).toHaveCount(0);
  }
});

test("a classification tab tallies its classes, and is read by accuracy", async ({ page }) => {
  // #185. The seeded PLS-DA sits below the ten-fold split beside the PLS, on
  // `fat_class` - Tecator's fat above its median. Every regression panel
  // applies, because the model is PLS1 on a dummy response; what differs is
  // the confusion panel where predicted-vs-measured would be, and the header.
  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /PLS-DA 5 LV/ }).first().dblclick();

  const header = page.getByTestId("analysis-header");
  await expect(header).toContainText("PLS-DA on fat_class 5 components");
  await expect(header).toContainText("ACCURACY (CV)");

  await expect(page.getByRole("region", { name: "Confusion" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Predicted vs measured" })).toHaveCount(0);
  for (const panel of ["Scores", "Loadings", "Variable importance", "RMSECV", "Calibration metrics"]) {
    await expect(page.getByRole("region", { name: panel })).toBeVisible();
  }

  // Fold zero's calibration tally sums to its rows, and the cross-validated
  // one to every sample; both are counts the server tallied, not the page.
  const totals = await page.evaluate(() => {
    const sum = (id: string) =>
      Array.from(document.querySelectorAll(`[data-testid=confusion-${id}] td.n`)).reduce(
        (total, cell) => total + Number(cell.textContent),
        0,
      );
    return { calibration: sum("calibration"), cv: sum("cross_validation") };
  });
  expect(totals).toEqual({ calibration: 216, cv: 240 });
  // #269: each set's per-class table, one row per class, read from its matrix.
  const perClass = page.getByTestId("class-metrics-cross_validation").locator("tbody tr");
  await expect(perClass).toHaveCount(2);
  await expect(perClass.first().locator("td").nth(1)).not.toHaveText("—");
  await expect(page.getByTestId("metric-Accuracy")).not.toHaveText("—");
  await expect(page.getByTestId("metric-Accuracy (CV)")).not.toHaveText("—");
});

test("picking an outlier draws which variables put it there, summing to its own T²", async ({
  page,
}) => {
  // #186. The diagnostics table lists what the limits put outside; a row
  // opens that sample's contributions, served by the node's own endpoint.
  await openResults(page);
  await expect(page.getByTestId("contributions-empty")).toBeVisible();

  const row = page.getByTestId("outlier-row").first();
  const index = Number(await row.getAttribute("data-index"));
  await row.click();
  await expect(page.getByTestId("contributions-plot")).toBeVisible();
  await expect(page.getByTestId("contributions-note")).toContainText("Σ =");

  const drawn = await page.evaluate(() => {
    const plot = document.querySelector("[data-testid=contributions-plot]") as HTMLElement & {
      data?: { x?: number[]; y?: number[] }[];
    };
    const trace = plot.data?.[0];
    return { points: trace?.x?.length ?? 0, sum: (trace?.y ?? []).reduce((a, b) => a + b, 0) };
  });
  expect(drawn.points).toBe(100);

  // The sum is the sample's T² as the results payload serves it, to the
  // store's float32 precision - the contributions are drawn, not decided.
  const served = await page.evaluate(async (wanted: number) => {
    const response = await fetch("/api/results/pca_a", {
      headers: { Authorization: `Bearer ${sessionStorage.getItem("token")}` },
    });
    const pca = await response.json();
    const position = pca.samples.findIndex((s: { index: number }) => s.index === wanted);
    return pca.diagnostics.hotelling_t2[position] as number;
  }, index);
  expect(Math.abs(drawn.sum - served) / served).toBeLessThan(1e-3);

  await page.getByLabel("Contribution view").selectOption("spe");
  await expect(page.getByTestId("contributions-plot")).toBeVisible();
  await expect(page.getByTestId("contributions-note")).toContainText("Σ =");
});

test("a SIMCA tab reads as class models: acceptance, Coomans and limits", async ({ page }) => {
  // #275. Added beside the seeded PLS-DA through the pipeline's PUT, read,
  // then the seeded recipe is put back: this project outlives the test.
  const auth = { Authorization: "Bearer e2e-token" };
  await page.goto("/?token=e2e-token");
  const original = (await (await page.request.get("/api/pipelines/current", { headers: auth })).json())
    .nodes as { id: string; inputs: string[] }[];
  const plsda = original.find((node) => node.id === "plsda_d")!;
  const nodes = [
    ...original,
    { id: "simca_d", type: "estimator", inputs: plsda.inputs, spec: { kind: "simca", n_components: 3, class_column: "fat_class" } },
  ];
  const put = (body: unknown[]) =>
    page.request.put("/api/pipelines/current", {
      headers: { ...auth, "Content-Type": "application/json" },
      data: { nodes: body },
    });
  expect((await put(nodes)).status()).toBe(200);
  try {
    await page.reload();
    await page.getByRole("button", { name: "Run pipeline" }).click();
    await expect(page.locator(".status")).toContainText("Done", { timeout: 120_000 });
    const outline = page.getByRole("complementary", { name: "Project outline" });
    await outline.getByRole("button", { name: /SIMCA 3 PC · fat_class/ }).dblclick();
    await expect(page.getByTestId("analysis-header")).toContainText(
      "SIMCA on fat_class 3 components per class",
    );
    await expect(page.getByTestId("acceptance-cross_validation").locator("tbody tr")).toHaveCount(2);
    await expect(page.getByTestId("coomans-plot")).toBeVisible();
    await expect(page.getByRole("region", { name: "Class models" })).toContainText("high");
    await expect(page.getByRole("region", { name: "Scores" })).toHaveCount(0);
  } finally {
    expect((await put(original)).status()).toBe(200);
  }
});

test("an LDA tab reads as a classification on the PCA front's scores", async ({ page }) => {
  // #276. As the SIMCA test does: added beside the seeded PLS-DA, then removed.
  const auth = { Authorization: "Bearer e2e-token" };
  await page.goto("/?token=e2e-token");
  const original = (await (await page.request.get("/api/pipelines/current", { headers: auth })).json())
    .nodes as { id: string; inputs: string[] }[];
  const plsda = original.find((node) => node.id === "plsda_d")!;
  const put = (body: unknown[]) =>
    page.request.put("/api/pipelines/current", {
      headers: { ...auth, "Content-Type": "application/json" },
      data: { nodes: body },
    });
  const lda = { id: "lda_d", type: "estimator", inputs: plsda.inputs, spec: { kind: "lda", n_components: 5, class_column: "fat_class" } };
  expect((await put([...original, lda])).status()).toBe(200);
  try {
    await page.reload();
    await page.getByRole("button", { name: "Run pipeline" }).click();
    await expect(page.locator(".status")).toContainText("Done", { timeout: 120_000 });
    const outline = page.getByRole("complementary", { name: "Project outline" });
    await outline.getByRole("button", { name: /LDA 5 PC · fat_class/ }).dblclick();
    await expect(page.getByTestId("analysis-header")).toContainText("LDA on fat_class 5 components");
    await expect(page.getByTestId("confusion-cross_validation")).toBeVisible();
    await expect(page.getByTestId("metric-Accuracy (CV)")).not.toHaveText("—");
    await expect(page.getByRole("region", { name: "Scores" })).toBeVisible();
  } finally {
    expect((await put(original)).status()).toBe(200);
  }
});

test("a kNN tab reads as a classification, and exports its JSON model", async ({ page }) => {
  // #277. As the SIMCA and LDA tests do: added beside the seeded PLS-DA, then removed.
  const auth = { Authorization: "Bearer e2e-token" };
  await page.goto("/?token=e2e-token");
  const original = (await (await page.request.get("/api/pipelines/current", { headers: auth })).json())
    .nodes as { id: string; inputs: string[] }[];
  const plsda = original.find((node) => node.id === "plsda_d")!;
  const put = (body: unknown[]) =>
    page.request.put("/api/pipelines/current", {
      headers: { ...auth, "Content-Type": "application/json" },
      data: { nodes: body },
    });
  const knn = {
    id: "knn_d",
    type: "estimator",
    inputs: plsda.inputs,
    spec: { kind: "knn", k: 5, n_components: 5, class_column: "fat_class" },
  };
  expect((await put([...original, knn])).status()).toBe(200);
  try {
    await page.reload();
    await page.getByRole("button", { name: "Run pipeline" }).click();
    await expect(page.locator(".status")).toContainText("Done", { timeout: 120_000 });
    const outline = page.getByRole("complementary", { name: "Project outline" });
    await outline.getByRole("button", { name: /kNN k5 · fat_class/ }).dblclick();
    await expect(page.getByTestId("analysis-header")).toContainText("kNN on fat_class 5 components");
    await expect(page.getByTestId("confusion-cross_validation")).toBeVisible();
    // #306: exported with its neighbours behind the chain's affine map.
    const [file] = await Promise.all([page.waitForEvent("download"), page.getByTestId("export-json").click()]);
    expect(file.suggestedFilename()).toBe("knn_d_model.json");
    const model = JSON.parse(await readFile((await file.path())!, "utf8"));
    expect(model.model.assignment).toBe("knn");
    expect(model.knn.k).toBe(5);
  } finally {
    expect((await put(original)).status()).toBe(200);
  }
});

/** Hover a plot's point by its data coordinates, through Plotly's own axes. */
async function hoverPoint(page: Page, testId: string, x: number, y: number) {
  const plot = page.getByTestId(testId);
  await plot.scrollIntoViewIfNeeded();
  const box = (await plot.boundingBox())!;
  const at = await plot.evaluate(
    (element, point) => {
      const layout = (
        element as HTMLElement & {
          _fullLayout: {
            margin: { l: number; t: number };
            xaxis: { l2p: (v: number) => number };
            yaxis: { l2p: (v: number) => number };
          };
        }
      )._fullLayout;
      return {
        x: layout.margin.l + layout.xaxis.l2p(point.x),
        y: layout.margin.t + layout.yaxis.l2p(point.y),
      };
    },
    { x, y },
  );
  await page.mouse.move(box.x + at.x, box.y + at.y);
}

test("an outlier row flags samples by rule, and hovering one names it", async ({ page }) => {
  // #278, outliers.md section 5.
  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /PLS 5 LV/ }).first().dblclick();
  const served = await (
    await page.request.get("/api/results/pls_d", { headers: { Authorization: "Bearer e2e-token" } })
  ).json();
  const block = await (
    await page.request.get("/api/results/pls_d/outliers", { headers: { Authorization: "Bearer e2e-token" } })
  ).json();
  const flags = block.flags as { index: number; rules: string[] }[];
  expect(flags.length).toBeGreaterThan(0);

  const row = page.getByTestId("outliers-row");
  await row.scrollIntoViewIfNeeded();
  for (const panel of ["Influence", "Leverage vs residual", "Flagged samples"]) {
    await expect(row.getByRole("region", { name: panel })).toBeVisible();
  }
  await expect(page.getByTestId("flag-row")).toHaveCount(flags.length);
  const first = served.samples[flags[0].index].sample_id as string;
  await expect(page.getByTestId("flag-row").first()).toContainText(first);

  // DESIGN_BRIEF section 7: a point on the plot is named on hover. The one
  // furthest out, so no neighbour sits under the cursor.
  const { hotelling_t2: t2, spe, hotelling_t2_limit: t2Limit, spe_limit: qLimit } =
    served.diagnostics as { hotelling_t2: number[]; spe: number[]; hotelling_t2_limit: number; spe_limit: number };
  const far = (i: number) => t2[i] / t2Limit + spe[i] / qLimit;
  const index = flags.map((flag) => flag.index).reduce((a, b) => (far(b) > far(a) ? b : a));
  await hoverPoint(page, "influence-plot", t2[index], spe[index]);
  await expect(page.locator(".hovertext")).toContainText(served.samples[index].sample_id);

  // A flag asks for a look: the row opens the sample's contributions.
  await page.getByTestId("flag-row").first().click();
  await expect(page.getByTestId("flag-row").first()).toHaveAttribute("aria-pressed", "true");
});

test("a PCA's outlier row has no residual plot", async ({ page }) => {
  await openResults(page);
  const row = page.getByTestId("outliers-row");
  await row.scrollIntoViewIfNeeded();
  await expect(row.getByRole("region", { name: "Influence" })).toBeVisible();
  await expect(row.getByRole("region", { name: "Flagged samples" })).toBeVisible();
  await expect(page.getByTestId("leverage-plot")).toHaveCount(0);
});

test("a VIP selection is applied as a step above a copy of the PLS, which runs", async ({ page }) => {
  // #281. Restored afterwards, as the SIMCA, LDA and kNN tests restore theirs.
  const auth = { Authorization: "Bearer e2e-token" };
  await page.goto("/?token=e2e-token");
  const original = (await (await page.request.get("/api/pipelines/current", { headers: auth })).json())
    .nodes as unknown[];
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /PLS 5 LV/ }).first().dblclick();
  await page.getByLabel("Variable importance view").selectOption("selection");
  await expect(page.getByTestId("selection-plot")).toBeVisible();

  // VIP ≥ 1 by default; what it keeps is counted against the variables there are.
  const served = await (await page.request.get("/api/results/pls_d", { headers: auth })).json();
  const kept = (served.regression.vip as number[]).filter((value) => value >= 1).length;
  await expect(page.getByTestId("selection-count")).toHaveText(`${kept} of 100`);

  try {
    await page.getByRole("button", { name: "Apply selection" }).click();
    // The copy runs on what was kept, and nothing else.
    await expect
      .poll(
        async () => {
          const response = await page.request.get("/api/results/pls_d_selected", { headers: auth });
          return response.ok() ? (await response.json()).n_variables : 0;
        },
        { timeout: 120_000 },
      )
      .toBe(kept);
    const nodes = (await (await page.request.get("/api/pipelines/current", { headers: auth })).json())
      .nodes as { id: string; step?: { kind: string; indices?: number[] } }[];
    const step = nodes.find((node) => node.id === "pls_d_select")!.step!;
    expect(step.kind).toBe("select_variables");
    expect(step.indices).toHaveLength(kept);
  } finally {
    expect(
      (
        await page.request.put("/api/pipelines/current", {
          headers: { ...auth, "Content-Type": "application/json" },
          data: { nodes: original },
        })
      ).status(),
    ).toBe(200);
  }
});

test("iPLS runs on the PLS's folds, draws its intervals, and its selection is applied", async ({
  page,
}) => {
  // #282. Restored afterwards, as the VIP test restores its pipeline.
  const auth = { Authorization: "Bearer e2e-token" };
  await page.goto("/?token=e2e-token");
  const original = (await (await page.request.get("/api/pipelines/current", { headers: auth })).json())
    .nodes as unknown[];
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /PLS 5 LV/ }).first().dblclick();
  await page.getByLabel("Variable importance view").selectOption("selection");
  await page.getByLabel("Select by").selectOption("ipls");
  await page.getByLabel("Intervals").fill("10");
  await page.getByRole("button", { name: "Run iPLS" }).click();

  const served = await (
    await page.request.get("/api/results/pls_d/ipls?n_intervals=10", { headers: auth })
  ).json();
  const kept = (served.selected as number[]).length;
  await expect(page.getByTestId("selection-count")).toHaveText(`${kept} of 100`);
  // One bar per interval, drawn from what the server sent.
  await expect
    .poll(() =>
      page.evaluate(() => {
        const plot = document.querySelector("[data-testid=selection-plot]") as HTMLElement & {
          data?: { type?: string; y?: number[] }[];
        };
        return plot.data?.[0]?.type === "bar" ? (plot.data[0].y?.length ?? 0) : 0;
      }),
    )
    .toBe(10);

  try {
    await page.getByRole("button", { name: "Apply selection" }).click();
    await expect
      .poll(
        async () => {
          const response = await page.request.get("/api/results/pls_d_selected", { headers: auth });
          return response.ok() ? (await response.json()).n_variables : 0;
        },
        { timeout: 120_000 },
      )
      .toBe(kept);
  } finally {
    expect(
      (
        await page.request.put("/api/pipelines/current", {
          headers: { ...auth, "Content-Type": "application/json" },
          data: { nodes: original },
        })
      ).status(),
    ).toBe(200);
  }
});

test("CARS runs seeded on the PLS's folds, is applied, and the pipeline warns of the leak", async ({
  page,
}) => {
  // #283. Restored afterwards, as the iPLS test restores its pipeline.
  const auth = { Authorization: "Bearer e2e-token" };
  await page.goto("/?token=e2e-token");
  const original = (await (await page.request.get("/api/pipelines/current", { headers: auth })).json())
    .nodes as unknown[];
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /PLS 5 LV/ }).first().dblclick();
  await page.getByLabel("Variable importance view").selectOption("selection");
  await page.getByLabel("Select by").selectOption("cars");
  await page.getByLabel("Sampling runs").fill("20");
  await page.getByRole("button", { name: "Run CARS" }).click();

  // Seeded, so what the screen ran is what this request runs.
  const served = await (
    await page.request.get("/api/results/pls_d/cars?n_runs=20", { headers: auth })
  ).json();
  const kept = (served.selected as number[]).length;
  await expect(page.getByTestId("selection-count")).toHaveText(`${kept} of 100`);
  await expect
    .poll(() =>
      page.evaluate(() => {
        const plot = document.querySelector("[data-testid=selection-plot]") as HTMLElement & {
          data?: { x?: number[] }[];
        };
        return plot.data?.[0]?.x?.length ?? 0;
      }),
    )
    .toBe(20);

  try {
    await page.getByRole("button", { name: "Apply selection" }).click();
    await expect
      .poll(
        async () => {
          const response = await page.request.get("/api/results/pls_d_selected", { headers: auth });
          return response.ok() ? (await response.json()).n_variables : 0;
        },
        { timeout: 120_000 },
      )
      .toBe(kept);
    // The copy is validated on the samples that chose its variables.
    const checked = await (
      await page.request.post("/api/pipelines/current/validate", { headers: auth })
    ).json();
    const leak = (checked.warnings as { code: string; node_id: string; related: string[] }[]).find(
      (warning) => warning.code === "selection_shares_samples",
    );
    expect(leak?.node_id).toBe("pls_d_select");
    expect(leak?.related).toEqual(["pls_d_selected"]);
    await expect(outline.getByRole("button", { name: /Select \d+ vars · CARS/ })).toBeVisible();
  } finally {
    expect(
      (
        await page.request.put("/api/pipelines/current", {
          headers: { ...auth, "Content-Type": "application/json" },
          data: { nodes: original },
        })
      ).status(),
    ).toBe(200);
  }
});
