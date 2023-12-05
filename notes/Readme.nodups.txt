
    nodups is a bash script that will scan an input file for lines matching lines already
    read in. The original lines are written out to a file while the matching lines are
    ignored.

    Watchout for any lines containing control sequences like ^N because bash will pluck that character out of the line before passing it to the script. So, the script will write out a line with a missing control character.

    This is a bash behavior and while the script could be adapted to fix this issue the code would be tricky to make work across versions of bash, or sh.

