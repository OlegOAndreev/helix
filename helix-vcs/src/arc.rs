use anyhow::{bail, Context, Result};
use arc_swap::ArcSwap;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use crate::FileChange;

#[derive(Deserialize)]
struct ArcStatus {
    status: StatusCategories,
}

#[derive(Deserialize)]
struct StatusCategories {
    staged: Vec<StatusEntry>,
    changed: Vec<StatusEntry>,
    untracked: Vec<StatusEntry>,
}

#[derive(Deserialize)]
struct StatusEntry {
    status: String,
    #[serde(rename = "type")]
    entry_type: String,
    path: String,
}

#[derive(Deserialize)]
struct BranchInfo {
    name: String,
    current: Option<bool>,
    #[allow(dead_code)]
    local: bool,
}

pub fn get_diff_base(file: &Path) -> Result<Vec<u8>> {
    debug_assert!(!file.exists() || file.is_file());
    debug_assert!(file.is_absolute());
    let file = file.canonicalize().context("resolve symlinks")?;

    let repo_dir = get_repo_dir(&file)?;

    let root_output =
        run_arc_command(&["root"], &repo_dir).context("failed to get arc repository root")?;
    let repo_root = PathBuf::from(root_output.trim());

    let rel_path = file
        .strip_prefix(&repo_root)
        .context("file not in repository")?;

    let path_arg = PathBuf::from("HEAD").join(rel_path);
    let path_str = path_arg.to_str().context("path contains invalid UTF-8")?;

    let output =
        run_arc_command(&["show", path_str], &repo_root).context("failed to get diff base")?;

    Ok(output.into_bytes())
}

pub fn get_current_head_name(file: &Path) -> Result<Arc<ArcSwap<Box<str>>>> {
    debug_assert!(!file.exists() || file.is_file());
    debug_assert!(file.is_absolute());
    let file = file.canonicalize().context("resolve symlinks")?;

    let repo_dir = get_repo_dir(&file)?;

    let output = run_arc_command(&["branch", "--json"], &repo_dir)
        .context("failed to get branch information")?;

    let branches: Vec<BranchInfo> =
        serde_json::from_str(&output).context("failed to parse branch JSON")?;

    let name = match branches.iter().find(|branch| branch.current == Some(true)) {
        Some(branch) => branch.name.clone(),
        None => String::new(),
    };

    Ok(Arc::new(ArcSwap::from_pointee(name.into_boxed_str())))
}

pub fn for_each_changed_file(cwd: &Path, f: impl Fn(Result<FileChange>) -> bool) -> Result<()> {
    let output = run_arc_command(&["root"], cwd).context("failed to get arc repository root")?;
    let repo_root = PathBuf::from(output.trim());

    let output = run_arc_command(&["status", "--json"], cwd).context("failed to get arc status")?;

    let status: ArcStatus =
        serde_json::from_str(&output).context("failed to parse arc status JSON")?;

    for entry in status
        .status
        .staged
        .iter()
        .chain(status.status.changed.iter())
    {
        if entry.entry_type == "directory" {
            continue;
        }

        let change = parse_status_entry(entry, &repo_root)?;
        if !f(Ok(change)) {
            return Ok(());
        }
    }

    for entry in &status.status.untracked {
        if entry.entry_type == "directory" {
            continue;
        }

        let change = parse_status_entry(entry, &repo_root)?;
        if !f(Ok(change)) {
            return Ok(());
        }
    }

    Ok(())
}

fn parse_status_entry(entry: &StatusEntry, repo_dir: &Path) -> Result<FileChange> {
    let path = repo_dir.join(&entry.path);

    match entry.status.as_str() {
        "new file" | "modified" => Ok(FileChange::Modified { path }),
        "untracked" => Ok(FileChange::Untracked { path }),
        "deleted" => Ok(FileChange::Deleted { path }),
        "conflict" | "conflicted" => Ok(FileChange::Conflict { path }),
        "renamed" => Ok(FileChange::Renamed {
            from_path: repo_dir.join(&entry.path),
            to_path: path,
        }),
        status => bail!("Unknown Arc status: {}", status),
    }
}

fn get_repo_dir(file: &Path) -> Result<&Path> {
    file.parent().context("file has no parent directory")
}

fn run_arc_command(args: &[&str], cwd: &Path) -> Result<String> {
    let output = Command::new("arc")
        .args(args)
        .current_dir(cwd)
        .output()
        .context("failed to execute arc command")?;

    if output.status.success() {
        let stdout =
            String::from_utf8(output.stdout).context("arc output contains invalid UTF-8")?;
        Ok(stdout)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "arc command failed with status {}: {}",
            output.status,
            stderr
        );
    }
}
