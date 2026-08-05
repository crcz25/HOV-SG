#!/usr/bin/env bash
#
# Prepare HM3DSEM scenes for HOV-SG in two stages, one scene at a time:
#
#   1. walks - render posed RGB / depth / semantic / pose frames with Habitat-Sim
#              (hovsg/data/hm3dsem/gen_hm3dsem_walks_from_poses.py)
#   2. gt    - compile floor-, region- and object-level ground truth
#              (hovsg/data/hm3dsem/create_hm3dsem_walks_gt.py)
#
# The eight scenes listed in the README carry manually curated trajectories and
# annotations and are handled by a separate workflow, so they are excluded from
# bulk discovery.  Naming one of them explicitly still processes it.
#
# Scenes are processed individually rather than in one batch call so that a
# per-scene failure (notably an out-of-memory kill during ground-truth
# compilation) is attributable, does not abort the remaining scenes, and is
# reported in the final summary.

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)

# ---------------------------------------------------------------------------
# Defaults - all repository-relative, none machine specific.
# ---------------------------------------------------------------------------
dataset_dir="$repo_dir/data/hm3d"
walks_dir="$repo_dir/data/hm3dsem_walks"
pose_dir="$repo_dir/hovsg/data/hm3dsem/metadata/poses"
floor_metadata="$repo_dir/hovsg/data/hm3dsem/metadata/Per_Scene_Floor_Sep.csv"
region_votes="$repo_dir/hovsg/data/hm3dsem/metadata/Per_Scene_Region_Weighted_Votes.csv"
region_labels="$repo_dir/hovsg/data/hm3dsem/metadata/Per_Scene_Region_Labels.csv"
walk_script="$repo_dir/hovsg/data/hm3dsem/gen_hm3dsem_walks_from_poses.py"
gt_script="$repo_dir/hovsg/data/hm3dsem/create_hm3dsem_walks_gt.py"
graph_config="$repo_dir/config/create_graph.yaml"

scene_config=""
generated_pose_dir=""
log_dir="$repo_dir/outputs/hm3dsem_preparation"
split="val"
stage="all"
python_bin="python"
conda_env=""
# gt_skip_frames in config/create_graph.yaml is not read by any Python code; the
# real ground-truth stride is --frame-step, and 1 keeps every recorded frame.
frame_step=1
max_frames=""
force=false
dry_run=false
process_all=false
include_excluded=false
requested_scenes=()

# The README evaluation scenes.  Kept identical to EXCLUDED_TRAJECTORY_SCENES in
# gen_hm3dsem_walks_from_poses.py; the check below fails loudly if they drift.
excluded_scenes=(
  00824-Dd4bFSTQ8gi
  00829-QaLdnwvtxbs
  00843-DYehNKdT76V
  00861-GLAQ4DNUx5U
  00862-LT9Jq6dN3Ea
  00873-bxsVRursffK
  00877-4ok3usBNeis
  00890-6s7QHgap2fW
)

usage() {
  cat <<'EOF'
Usage: scripts/process_hm3dsem_excluding_readme_eval.sh [options] [SCENE_ID ...]

Render posed RGB-D walks and compile HM3DSEM ground truth, one scene at a time.

Scene selection (choose one):
  SCENE_ID ...             Process exactly these scenes.  README evaluation
                           scenes are processed when named explicitly.
  --scene ID               Same as a positional ID; repeatable.
  --all                    Process every eligible scene in the split, minus the
                           eight README evaluation scenes.  This is the default
                           when no scene is named.
  --include-excluded       With --all, also process the eight README evaluation
                           scenes, so one run covers every eligible scene.

Options:
  --dataset-dir PATH       Raw HM3D dataset root (default: data/hm3d).
  --walks-dir PATH         hm3dsem_walks output root (default: data/hm3dsem_walks).
  --split NAME             Split directory under both roots (default: val).
  --pose-dir PATH          Read-only supplied <scene-id>.txt trajectories.
  --generated-pose-dir P   Where synthesized trajectories are written
                           (default: <walks-dir>/<split>/<scene>/trajectory.txt).
  --scene-config PATH      Explicit Habitat *.scene_dataset_config.json.
  --floor-metadata PATH    Floor-boundary CSV.
  --region-votes PATH      Region-vote CSV.
  --region-labels PATH     Manual region-label CSV.
  --frame-step N           Ground-truth frame stride (default: 1 = every frame).
  --max-frames N           Cap synthesized trajectories at N poses (renderer
                           default: 5000).  Supplied trajectories are unaffected.
  --stage NAME             all (default), walks, or gt.
  --python PATH            Python interpreter (default: python).
  --conda-env NAME         Activate this conda environment first.
  --log-dir PATH           Per-scene stage logs (default: outputs/hm3dsem_preparation).
  --force                  Regenerate outputs that already validate as complete.
  --dry-run                Validate inputs and report the plan; write nothing.
  -h, --help               Show this help.

Exit status is nonzero if any requested scene fails.

Examples:
  # One scene (a README evaluation scene, named explicitly)
  scripts/process_hm3dsem_excluding_readme_eval.sh 00824-Dd4bFSTQ8gi

  # A selected list of scenes
  scripts/process_hm3dsem_excluding_readme_eval.sh 00800-TEEsavR23oF 00802-wcojb4TFT35

  # All supported validation scenes, minus the eight README evaluation scenes
  scripts/process_hm3dsem_excluding_readme_eval.sh --all
EOF
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 2; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir)        [[ $# -ge 2 ]] || die "$1 needs a value"; dataset_dir=$2; shift 2 ;;
    --walks-dir)          [[ $# -ge 2 ]] || die "$1 needs a value"; walks_dir=$2; shift 2 ;;
    --split)              [[ $# -ge 2 ]] || die "$1 needs a value"; split=$2; shift 2 ;;
    --pose-dir)           [[ $# -ge 2 ]] || die "$1 needs a value"; pose_dir=$2; shift 2 ;;
    --generated-pose-dir) [[ $# -ge 2 ]] || die "$1 needs a value"; generated_pose_dir=$2; shift 2 ;;
    --scene-config)       [[ $# -ge 2 ]] || die "$1 needs a value"; scene_config=$2; shift 2 ;;
    --floor-metadata)     [[ $# -ge 2 ]] || die "$1 needs a value"; floor_metadata=$2; shift 2 ;;
    --region-votes)       [[ $# -ge 2 ]] || die "$1 needs a value"; region_votes=$2; shift 2 ;;
    --region-labels)      [[ $# -ge 2 ]] || die "$1 needs a value"; region_labels=$2; shift 2 ;;
    --frame-step)         [[ $# -ge 2 ]] || die "$1 needs a value"; frame_step=$2; shift 2 ;;
    --max-frames)         [[ $# -ge 2 ]] || die "$1 needs a value"; max_frames=$2; shift 2 ;;
    --stage)              [[ $# -ge 2 ]] || die "$1 needs a value"; stage=$2; shift 2 ;;
    --python)             [[ $# -ge 2 ]] || die "$1 needs a value"; python_bin=$2; shift 2 ;;
    --conda-env)          [[ $# -ge 2 ]] || die "$1 needs a value"; conda_env=$2; shift 2 ;;
    --log-dir)            [[ $# -ge 2 ]] || die "$1 needs a value"; log_dir=$2; shift 2 ;;
    --scene)              [[ $# -ge 2 ]] || die "$1 needs a value"; requested_scenes+=("$2"); shift 2 ;;
    --all)                process_all=true; shift ;;
    --include-excluded)   include_excluded=true; shift ;;
    --force|--overwrite)  force=true; shift ;;
    --dry-run)            dry_run=true; shift ;;
    -h|--help)            usage; exit 0 ;;
    --) shift; while [[ $# -gt 0 ]]; do requested_scenes+=("$1"); shift; done ;;
    -*) die "Unknown option: $1" ;;
    *)  requested_scenes+=("$1"); shift ;;
  esac
done

case "$stage" in
  all|walks|gt) ;;
  *) die "--stage must be one of: all, walks, gt" ;;
esac
[[ "$frame_step" =~ ^[1-9][0-9]*$ ]] || die "--frame-step must be a positive integer, got: $frame_step"
if [[ -n "$max_frames" ]]; then
  [[ "$max_frames" =~ ^[1-9][0-9]*$ ]] || die "--max-frames must be a positive integer, got: $max_frames"
  [[ "$max_frames" -ge 2 ]] || die "--max-frames must be at least 2"
fi
if [[ ${#requested_scenes[@]} -gt 0 && "$process_all" == true ]]; then
  die "--all cannot be combined with explicit scene IDs"
fi
[[ ${#requested_scenes[@]} -eq 0 ]] && process_all=true

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
if [[ -n "$conda_env" ]]; then
  command -v conda >/dev/null 2>&1 || die "--conda-env given but conda is not on PATH"
  conda_base=$(conda info --base) || die "could not resolve the conda base prefix"
  [[ -f "$conda_base/etc/profile.d/conda.sh" ]] || die "conda.sh not found under $conda_base"
  # conda's shell hook is not written for `set -u`/`set -e`.
  set +eu
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$conda_env"
  activation_status=$?
  set -eu
  [[ $activation_status -eq 0 ]] || die "could not activate conda environment: $conda_env"
  log "Activated conda environment: $conda_env"
fi

command -v "$python_bin" >/dev/null 2>&1 || die "Python interpreter not found: $python_bin"
python_bin=$(command -v "$python_bin")

# create_hm3dsem_walks_gt.py imports `scripts.generate_hm3dsem_semantic_label_csv`,
# so the repository root must be importable regardless of the caller's cwd.
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export HABITAT_SIM_LOG="${HABITAT_SIM_LOG:-quiet}"

# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------
log "Validating dependencies and inputs"

[[ -f "$walk_script" ]] || die "Walk generator not found: $walk_script"
[[ -f "$gt_script" ]]   || die "Ground-truth generator not found: $gt_script"
[[ -d "$dataset_dir" ]] || die "Dataset directory does not exist: $dataset_dir"

split_dir="$dataset_dir/$split"
[[ -d "$split_dir" ]] || die "Split directory does not exist: $split_dir"

if ! "$python_bin" -c 'import habitat_sim, numpy, open3d, cv2, pandas, scipy' >/dev/null 2>&1; then
  "$python_bin" -c 'import habitat_sim, numpy, open3d, cv2, pandas, scipy' || true
  die "Required Python packages are missing (habitat_sim, numpy, open3d, cv2, pandas, scipy)"
fi

# The Habitat scene-dataset config is required by both stages.
if [[ -n "$scene_config" ]]; then
  [[ -f "$scene_config" ]] || die "Scene dataset config does not exist: $scene_config"
else
  shopt -s nullglob
  config_candidates=("$dataset_dir"/*.scene_dataset_config.json)
  shopt -u nullglob
  [[ ${#config_candidates[@]} -gt 0 ]] || die "No *.scene_dataset_config.json found in $dataset_dir"
fi

[[ -d "$pose_dir" ]] || warn "Pose directory does not exist: $pose_dir (all trajectories will be synthesized)"
for optional in "$floor_metadata" "$region_votes" "$region_labels"; do
  [[ -f "$optional" ]] || warn "Optional metadata CSV not found, continuing without it: $optional"
done

# The exclusion list must not drift from the renderer's copy.
python_excluded=$("$python_bin" - <<'PY'
from hovsg.data.hm3dsem.gen_hm3dsem_walks_from_poses import EXCLUDED_TRAJECTORY_SCENES
print(" ".join(sorted(EXCLUDED_TRAJECTORY_SCENES)))
PY
) || die "Could not import the renderer's exclusion list; check PYTHONPATH and the conda environment"
shell_excluded=$(printf '%s\n' "${excluded_scenes[@]}" | sort | tr '\n' ' ' | sed 's/ $//')
[[ "$python_excluded" == "$shell_excluded" ]] || die \
  "Exclusion list drift. Renderer: [$python_excluded] / this script: [$shell_excluded]"

# config/create_graph.yaml belongs to the scene-graph stage; this workflow must
# not touch it.  Record a checksum so an accidental edit is reported at exit.
graph_config_checksum=""
if [[ -f "$graph_config" ]]; then
  graph_config_checksum=$(sha256sum "$graph_config" | cut -d' ' -f1)
fi

check_graph_config() {
  [[ -n "$graph_config_checksum" && -f "$graph_config" ]] || return 0
  local now
  now=$(sha256sum "$graph_config" | cut -d' ' -f1)
  if [[ "$now" != "$graph_config_checksum" ]]; then
    warn "$graph_config changed during this run; restore it before committing."
  fi
}
trap check_graph_config EXIT

# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------
is_excluded() {
  local candidate=$1 scene
  for scene in "${excluded_scenes[@]}"; do
    [[ "$scene" == "$candidate" ]] && return 0
  done
  return 1
}

# Mesh stem for a scene, e.g. 00824-Dd4bFSTQ8gi -> Dd4bFSTQ8gi.
scene_mesh_name() {
  local scene_dir=$1 meshes=()
  shopt -s nullglob
  meshes=("$scene_dir"/*.basis.glb)
  shopt -u nullglob
  if [[ ${#meshes[@]} -eq 1 ]]; then
    basename -- "${meshes[0]}" .basis.glb
    return 0
  fi
  return 1
}

# Echo a human-readable reason when a scene cannot be processed.
raw_scene_problem() {
  local scene=$1 scene_dir="$split_dir/$scene" name
  [[ -d "$scene_dir" ]] || { echo "raw scene directory is missing: $scene_dir"; return 0; }
  if ! name=$(scene_mesh_name "$scene_dir"); then
    echo "expected exactly one *.basis.glb in $scene_dir"
    return 0
  fi
  local required=("$scene_dir/$name.basis.glb" "$scene_dir/$name.semantic.glb" "$scene_dir/$name.semantic.txt")
  local pose_file="$pose_dir/$scene.txt" path
  # Without a supplied trajectory the renderer plans one from the navmesh.
  [[ -f "$pose_file" ]] || required+=("$scene_dir/$name.basis.navmesh")
  for path in "${required[@]}"; do
    [[ -f "$path" ]] || { echo "missing required source file: $path"; return 0; }
  done
  if [[ -f "$pose_file" ]]; then
    # One whitespace-separated 4x4 matrix per line, all sixteen fields finite.
    # gen_hm3dsem_walks_from_poses.py re-validates; this only fails fast.
    if ! awk '
        NF==0 { next }
        NF!=16 { exit 1 }
        { for (i = 1; i <= 16; i++) {
            # Compare $i itself: assigning it to a variable would drop awks
            # numeric-string attribute and turn this into a string compare.
            if (tolower($i) ~ /nan|inf/) exit 1
            if ($i + 0 != $i) exit 1
          }
          rows++ }
        END { exit (rows == 0) }' "$pose_file"; then
      echo "malformed trajectory pose file (expected 16 finite values per line): $pose_file"
      return 0
    fi
  elif is_excluded "$scene"; then
    echo "README evaluation scene without a supplied trajectory: $pose_file"
    return 0
  fi
  echo ""
}

# Which trajectory the renderer will use.  Mirrors should_generate_trajectory()
# in gen_hm3dsem_walks_from_poses.py: a file under --pose-dir always wins.
trajectory_source() {
  if [[ -f "$pose_dir/$1.txt" ]]; then
    echo "supplied ($pose_dir/$1.txt)"
  else
    echo "synthesized from navmesh"
  fi
}

# Sorted frame stems for a modality directory.
frame_stems() {
  local dir=$1 ext=$2
  [[ -d "$dir" ]] || return 1
  find "$dir" -maxdepth 1 -type f -name "*.$ext" -printf '%f\n' | sed "s/\.${ext}\$//" | sort
}

walk_complete() {
  local scene=$1 out="$walks_dir/$split/$scene" tmp count kind ok
  local -a kinds=(rgb:png depth:png semantic:npy pose:txt)
  for kind in "${kinds[@]}"; do
    [[ -d "$out/${kind%%:*}" ]] || return 1
  done
  # A walk is complete when it holds one frame per pose of the trajectory it
  # was rendered from.  Comparing against the trajectory catches an interrupted
  # run, and accepts a walk rendered by an earlier version whose frames are all
  # present -- a supplied trajectory is rendered in full, so the count is exact.
  local expected="" traj=""
  if [[ -f "$pose_dir/$scene.txt" ]]; then
    traj="$pose_dir/$scene.txt"
  elif [[ -f "$out/trajectory.txt" ]]; then
    traj="$out/trajectory.txt"
  fi
  if [[ -n "$traj" ]]; then
    expected=$(grep -cve '^[[:space:]]*$' "$traj" || true)
  elif [[ ! -f "$out/camera_info.json" ]]; then
    # No trajectory to compare against and no renderer marker: treat as partial.
    return 1
  fi
  tmp=$(mktemp) || return 1
  frame_stems "$out/rgb" png >"$tmp" || { rm -f "$tmp"; return 1; }
  count=$(wc -l <"$tmp")
  if [[ "$count" -eq 0 ]]; then rm -f "$tmp"; return 1; fi
  if [[ -n "$expected" && "$count" -ne "$expected" ]]; then rm -f "$tmp"; return 1; fi
  ok=0
  for kind in depth:png semantic:npy pose:txt; do
    if ! frame_stems "$out/${kind%%:*}" "${kind##*:}" | cmp -s - "$tmp"; then ok=1; break; fi
  done
  rm -f "$tmp"
  return $ok
}

gt_complete() {
  local scene=$1 out="$walks_dir/$split/$scene" path
  for path in objects regions semantic_label_map.csv scene_info.json scene_panoptic.ply scene_rgb.ply; do
    [[ -e "$out/$path" ]] || return 1
  done
  # objects/ and regions/ are created empty by the generator before it writes.
  compgen -G "$out/objects/*.ply" >/dev/null || return 1
  compgen -G "$out/regions/*.ply" >/dev/null || return 1
  return 0
}

# ---------------------------------------------------------------------------
# Scene selection
# ---------------------------------------------------------------------------
declare -a scenes=()
declare -a skipped=() succeeded=() failed=()
declare -A skip_reason=()

if [[ "$process_all" == true ]]; then
  if [[ "$include_excluded" == true ]]; then
    log "Discovering scenes under $split_dir (including the README evaluation scenes)"
  else
    log "Discovering scenes under $split_dir (excluding ${#excluded_scenes[@]} README evaluation scenes)"
  fi
  excluded_seen=0
  while IFS= read -r scene_dir; do
    scene=$(basename -- "$scene_dir")
    if [[ "$include_excluded" != true ]] && is_excluded "$scene"; then
      excluded_seen=$((excluded_seen + 1))
      continue
    fi
    problem=$(raw_scene_problem "$scene")
    if [[ -n "$problem" ]]; then
      # Not every HM3D scene ships semantic annotations; that is expected here.
      skipped+=("$scene")
      skip_reason["$scene"]="ineligible: $problem"
      continue
    fi
    scenes+=("$scene")
  done < <(find "$split_dir" -mindepth 1 -maxdepth 1 -type d | sort)
  if [[ "$include_excluded" == true ]]; then
    log "Included the ${#excluded_scenes[@]} README evaluation scenes (--include-excluded)"
  else
    log "Excluded $excluded_seen README evaluation scene(s): ${excluded_scenes[*]}"
  fi
  log "Eligible: ${#scenes[@]} scene(s); ineligible: ${#skipped[@]} scene(s)"
else
  for scene in "${requested_scenes[@]}"; do
    if is_excluded "$scene"; then
      warn "$scene is a README evaluation scene; processing it because it was named explicitly."
    fi
    problem=$(raw_scene_problem "$scene")
    if [[ -n "$problem" ]]; then
      # Explicitly requested scenes must not be silently downgraded to a skip.
      failed+=("$scene")
      skip_reason["$scene"]="ineligible: $problem"
      warn "$scene cannot be processed: $problem"
      continue
    fi
    scenes+=("$scene")
  done
fi

if [[ ${#scenes[@]} -eq 0 && ${#failed[@]} -eq 0 ]]; then
  log "No scenes to process."
  exit 0
fi

# Make the trajectory split explicit before any rendering starts.
supplied_count=0
for scene in ${scenes[@]+"${scenes[@]}"}; do
  [[ -f "$pose_dir/$scene.txt" ]] && supplied_count=$((supplied_count + 1))
done
log "Trajectories: $supplied_count supplied from $pose_dir, $(( ${#scenes[@]} - supplied_count )) synthesized from the navmesh"

# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------
mkdir -p "$log_dir"

build_walk_command() {
  local scene=$1
  walk_command=(
    "$python_bin" "$walk_script"
    --dataset-dir "$dataset_dir"
    --save-dir "$walks_dir"
    --pose-dir "$pose_dir"
    --split "$split"
    --scene-id "$scene"
  )
  [[ -n "$scene_config" ]]        && walk_command+=(--scene-config "$scene_config")
  [[ -f "$floor_metadata" ]]      && walk_command+=(--floor-metadata "$floor_metadata")
  [[ -n "$generated_pose_dir" ]]  && walk_command+=(--generated-pose-dir "$generated_pose_dir")
  [[ -n "$max_frames" ]]          && walk_command+=(--max-trajectory-poses "$max_frames")
  # The renderer refuses to touch an existing scene directory without this.
  walk_command+=(--overwrite)
  return 0
}

build_gt_command() {
  local scene=$1
  gt_command=(
    "$python_bin" "$gt_script"
    --dataset-dir "$dataset_dir"
    --walks-dir "$walks_dir"
    --split "$split"
    --scene-id "$scene"
    --frame-step "$frame_step"
  )
  [[ -n "$scene_config" ]]   && gt_command+=(--scene-config "$scene_config")
  [[ -f "$floor_metadata" ]] && gt_command+=(--floor-metadata "$floor_metadata")
  [[ -f "$region_votes" ]]   && gt_command+=(--region-votes "$region_votes")
  [[ -f "$region_labels" ]]  && gt_command+=(--region-labels "$region_labels")
  return 0
}

# Ground-truth compilation accumulates a full-scene point cloud and needs on the
# order of 128 GB for large scenes, so an OOM kill is a likely failure mode and
# is worth naming rather than reporting as a generic nonzero exit.
describe_failure() {
  local status=$1 logfile=$2
  if [[ $status -eq 137 ]] || grep -qiE 'MemoryError|std::bad_alloc|Cannot allocate memory|Killed process|out of memory' "$logfile"; then
    echo "exit $status - out of memory (ground-truth compilation may need ~128 GB for large scenes)"
  elif [[ $status -eq 130 ]]; then
    echo "exit $status - interrupted"
  else
    echo "exit $status"
  fi
}

run_stage() {
  local scene=$1 stage_name=$2 logfile status
  shift 2
  logfile="$log_dir/${scene}-${stage_name}.log"
  log "  [$scene] $stage_name: running (log: $logfile)"
  set +e
  "$@" 2>&1 | tee "$logfile"
  status=${PIPESTATUS[0]}
  set -e
  if [[ $status -ne 0 ]]; then
    warn "[$scene] $stage_name failed: $(describe_failure "$status" "$logfile")"
    return 1
  fi
  return 0
}

if [[ "$dry_run" == true ]]; then
  log "Dry run: no outputs will be written."
  for scene in "${scenes[@]}"; do
    walk_state="pending"; walk_complete "$scene" && walk_state="complete"
    gt_state="pending";   gt_complete "$scene"   && gt_state="complete"
    log "  $scene: trajectory=$(trajectory_source "$scene") walks=$walk_state gt=$gt_state"
  done
  log "Would run stage '$stage' for ${#scenes[@]} scene(s) with --frame-step $frame_step."
  if [[ ${#scenes[@]} -gt 0 ]]; then
    build_walk_command "${scenes[0]}"; build_gt_command "${scenes[0]}"
    log "Example walk command: ${walk_command[*]}"
    log "Example gt   command: ${gt_command[*]}"
  fi
  [[ ${#failed[@]} -eq 0 ]] || exit 1
  exit 0
fi

total=${#scenes[@]}
index=0
for scene in "${scenes[@]}"; do
  index=$((index + 1))
  log "=== [$index/$total] $scene ==="
  scene_failed=false

  if [[ "$stage" == "all" || "$stage" == "walks" ]]; then
    if [[ "$force" != true ]] && walk_complete "$scene"; then
      log "  [$scene] walks: existing output validates as complete; skipping (use --force to regenerate)"
    else
      if ! walk_complete "$scene" && [[ -d "$walks_dir/$split/$scene" ]]; then
        warn "[$scene] existing walk output is incomplete (frame count does not match its trajectory); re-rendering"
      fi
      build_walk_command "$scene"
      if ! run_stage "$scene" walks "${walk_command[@]}"; then
        scene_failed=true
      fi
    fi
  fi

  # The ground-truth stage reads the rendered frames, so it must not run after a
  # failed or missing render.
  if [[ "$scene_failed" != true && ( "$stage" == "all" || "$stage" == "gt" ) ]]; then
    if ! walk_complete "$scene"; then
      warn "[$scene] gt: skipped because the rendered walk is missing or incomplete"
      scene_failed=true
    elif [[ "$force" != true ]] && gt_complete "$scene"; then
      log "  [$scene] gt: existing ground truth validates as complete; skipping (use --force to regenerate)"
    else
      build_gt_command "$scene"
      if ! run_stage "$scene" gt "${gt_command[@]}"; then
        scene_failed=true
      fi
    fi
  fi

  if [[ "$scene_failed" == true ]]; then
    failed+=("$scene")
    log "  [$scene] FAILED"
  else
    succeeded+=("$scene")
    log "  [$scene] OK"
  fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
log "===== Summary (split=$split, stage=$stage) ====="
log "Succeeded: ${#succeeded[@]}"
for scene in ${succeeded[@]+"${succeeded[@]}"}; do log "  OK      $scene"; done
log "Skipped:   ${#skipped[@]}"
for scene in ${skipped[@]+"${skipped[@]}"}; do log "  SKIP    $scene (${skip_reason[$scene]})"; done
log "Failed:    ${#failed[@]}"
for scene in ${failed[@]+"${failed[@]}"}; do
  log "  FAIL    $scene${skip_reason[$scene]+ (${skip_reason[$scene]})}"
done

if [[ ${#failed[@]} -gt 0 ]]; then
  log "One or more requested scenes failed; see $log_dir for per-scene logs."
  exit 1
fi
log "All requested scenes completed."
exit 0
