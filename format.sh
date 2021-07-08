#!/usr/bin/env bash
# Usage: at the root dir >> bash scripts/format.sh .

# Check yapf version. (20200318 latest is 0.29.0. Format might be changed in future version.)
ver=$(yapf --version)
if ! echo $ver | grep -q 0.29.0; then
  echo "Wrong YAPF version installed: 0.29.0 is required, not $ver. $YAPF_DOWNLOAD_COMMAND_MSG"
  exit 1
fi

yapf --in-place --recursive -p --verbose --style .style.yapf $1

if [[ "$2" == '--test' ]]; then # Only for CI usage, user should not use --test flag.
  if ! git diff --quiet &>/dev/null; then
    echo '*** You have not reformatted your codes! Please run [bash format.sh] at root directory before commit! Thanks! ***'
    exit 1
  else
    echo "Code style test passed!"
  fi
fi
