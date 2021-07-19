#!/bin/bash

# This scripts counts the lines of code and comments in all source files
# and prints the results to the command line. It uses the commandline tool
# "cloc". You can either pass --loc, --comments or --percentage to show the
# respective values only.
# Some parts below need to be adapted to your project!

# Get the location of this script.
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

# Run cloc - this counts code lines, blank lines and comment lines
# for the specified languages. You will need to change this accordingly.
# For C++, you could use "C++,C/C++ Header" for example.
# We are only interested in the summary, therefore the tail -1
SUMMARY="$(cloc "${SCRIPT_DIR}" --include-lang="Python" --md | tail -1)"

# The $SUMMARY is one line of a markdown table and looks like this:
# SUM:|101|3123|2238|10783
# We use the following command to split it into an array.
IFS='|' read -r -a TOKENS <<< "$SUMMARY"

# Store the individual tokens for better readability.
NUMBER_OF_FILES=${TOKENS[1]}
COMMENT_LINES=${TOKENS[3]}
LINES_OF_CODE=${TOKENS[4]}

# To make the estimate of commented lines more accurate, we have to
# subtract any copyright header which is included in each file.
# For Fly-Pie, this header has the length of five lines.
# All dumb comments like those /////////// or those // ------------ 
# are also subtracted. As cloc does not count inline comments,
# the overall estimate should be rather conservative.
# Change the lines below according to your project.
DUMB_COMMENTS="$(grep -r -E '//////|// -----' "${SCRIPT_DIR}" | wc -l)"
COMMENT_LINES=$(($COMMENT_LINES - 5 * $NUMBER_OF_FILES - $DUMB_COMMENTS))

# Print all results if no arguments are given.
if [[ $# -eq 0 ]] ; then
  awk -v a=$LINES_OF_CODE \
      'BEGIN {printf "Lines of source code: %6.1fk\n", a/1000}'
  awk -v a=$COMMENT_LINES \
      'BEGIN {printf "Lines of comments:    %6.1fk\n", a/1000}'
  awk -v a=$COMMENT_LINES -v b=$LINES_OF_CODE \
      'BEGIN {printf "Comment Percentage:   %6.1f%\n", 100*a/b}'
  exit 0
fi

# Show lines of code if --loc is given.
if [[ $* == *--loc* ]]
then
  awk -v a=$LINES_OF_CODE \
      'BEGIN {printf "%.1fk\n", a/1000}'
fi

# Show lines of comments if --comments is given.
if [[ $* == *--comments* ]]
then
  awk -v a=$COMMENT_LINES \
      'BEGIN {printf "%.1fk\n", a/1000}'
fi

# Show precentage of comments if --percentage is given.
if [[ $* == *--percentage* ]]
then
  awk -v a=$COMMENT_LINES -v b=$LINES_OF_CODE \
      'BEGIN {printf "%.1f\n", 100*a/b}'
fi

