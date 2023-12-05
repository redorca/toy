
    dedup.py will scan an input file looking for duplication of any lines already
    read in and ignoring them when writing an output file.

    python dedup.py -I <"path to source file" -O <"path to output (destination) file">

    Note that lines that differ only in a trailing space or two are considered
    duplicates of each other.

    E.G. These two lines are the same:
        "cd "
        "cd"

    When the line is written to the outpout file those extra spaces at the end of the
    line are removed.

