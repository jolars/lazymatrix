import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const config = await readFile(
  new URL("../versionary.jsonc", import.meta.url),
  "utf8",
);
const cases = [
  {
    name: "Python-only fix",
    files: ["python/api.py"],
    message: "fix: Python columns",
    rust: undefined,
    python: "0.1.1",
  },
  {
    name: "Rust feature",
    files: ["src/lib.rs"],
    message: "feat: Rust products",
    rust: "0.8.0",
    python: "0.2.0",
  },
  {
    name: "Combined fix",
    files: ["src/lib.rs", "python/api.py"],
    message: "fix: normalized products",
    rust: "0.7.1",
    python: "0.1.1",
  },
  {
    name: "First Python release",
    initial: "0.0.0",
    files: ["python/api.py"],
    message: "feat: Python bindings",
    rust: undefined,
    python: "0.1.0",
  },
  {
    name: "First combined release",
    initial: "0.0.0",
    files: ["src/lib.rs", "python/api.py"],
    message: "feat: Python bindings and Rust packaging",
    rust: "0.8.0",
    python: "0.1.0",
  },
];

for (const scenario of cases) {
  const directory = await mkdtemp(join(tmpdir(), "lazymatrix-release-"));
  const git = (...args) =>
    execFileSync("git", args, {
      cwd: directory,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    }).trim();
  const versionary = (...args) =>
    execFileSync("versionary", args, {
      cwd: directory,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "inherit"],
    });
  try {
    await mkdir(join(directory, "src"));
    await mkdir(join(directory, "python"));
    await writeFile(join(directory, "versionary.jsonc"), config);
    await writeFile(
      join(directory, "Cargo.toml"),
      '[package]\nname = "lazymatrix"\nversion = "0.7.0"\nedition = "2024"\n',
    );
    await writeFile(join(directory, "src/lib.rs"), "//! Fixture.\n");
    const initial = scenario.initial ?? "0.1.0";
    await writeFile(
      join(directory, "python/pyproject.toml"),
      `[project]\nname = "lazymatrix"\nversion = "${initial}"\n`,
    );
    await writeFile(join(directory, "python/api.py"), '"""Fixture."""\n');
    git("init", "-b", "main");
    git("config", "user.name", "Release configuration test");
    git("config", "user.email", "release-test@example.invalid");
    git("config", "commit.gpgsign", "false");
    git("config", "tag.gpgsign", "false");
    git("config", "core.hooksPath", join(directory, ".git/no-hooks"));
    git("add", ".");
    git("commit", "-m", "chore: initialize fixture");
    const baseline = git("rev-parse", "HEAD");
    git("tag", "v0.7.0");
    const targets = [{ path: ".", version: "0.7.0", tag: "v0.7.0" }];
    if (initial !== "0.0.0") {
      git("tag", `python-v${initial}`);
      targets.push({ path: "python", version: initial, tag: `python-v${initial}` });
    }
    await writeFile(
      join(directory, ".versionary-manifest.json"),
      JSON.stringify({
        "manifest-version": 1,
        "baseline-sha": baseline,
        "release-targets": targets,
      }),
    );
    git("add", ".");
    git("commit", "-m", "chore: record release baseline");
    for (const path of scenario.files) {
      await writeFile(
        join(directory, path),
        `${await readFile(join(directory, path), "utf8")}\n`,
      );
    }
    git("add", ".");
    git("commit", "-m", scenario.message);
    versionary("verify");
    const plan = JSON.parse(versionary("plan"));
    const rust = plan.packages.find((package_) => package_.path === ".");
    const python = plan.packages.find((package_) => package_.path === "python");
    assert.equal(rust.nextVersion, scenario.rust ?? null, scenario.name);
    assert.equal(python.nextVersion, scenario.python, scenario.name);
    assert.equal(plan.releaseBranchPrefix, "versionary/release");
    if (scenario.name === "Rust feature") {
      assert.equal(python.bumpReason, "follows");
    }
    const preview = JSON.parse(versionary("run", "--dry-run", "--json"));
    assert.equal(preview.action, "pr-dry-run");
    assert.equal(preview.branch, "versionary/release");
    const expected = [`python-v${scenario.python}`];
    if (scenario.rust) expected.push(`v${scenario.rust}`);
    assert.deepEqual(
      preview.targets.map((target) => target.tag).sort(),
      expected.sort(),
    );
    assert.equal(
      git("status", "--porcelain"),
      "",
      "Planning must not change release state",
    );
    console.log(`Checked ${scenario.name}.`);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}
