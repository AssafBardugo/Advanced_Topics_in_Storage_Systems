import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import operator
from typing import List


class TraceEntry(object):
    # Constructs an object from a string line read from input
    def __init__(self, line):
        time, disk, block, size, self.op = line.split()
        self.time = float(time)
        self.disk, self.block, self.size = int(disk), int(block), int(size)


class TraceData(object):
    def __init__(self, name):
        # Extract the name only from the path.
        # For example: from "submission_traces/syn1.trace" to "syn1".
        self.name = name.split('/')[-1].split('.')[0]

        self.unique_reads = 0           # The number of different blocks accessed for reading
        self.unique_writes = 0          # The number of different blocks accessed for writing
        self.unique_total = 0           # The number of different blocks accessed for reading or writing

        self.read_requests = 0          # Sum of read lines
        self.write_requests = 0         # Sum of write lines

        self.reads_volume = 0           # The total amount of blocks read
        self.writes_volume = 0          # The total amount of blocks write
    

# ======= Calculate the data of each trace into a TraceData object =======
def calculate_trace_data(trace: str) -> TraceData:

    td = TraceData(trace)

    # The following Lists will contain for each line in the 
    #  trace an interval of [first-block-number, last-block-number]
    read_intervals = []
    write_intervals = []
    
    with open(trace, 'r') as trace_fd:
        for line in trace_fd:
            te = TraceEntry(line)
            if te.op == 'R':
                td.read_requests += 1
                td.reads_volume += te.size
                read_intervals.append([te.block, te.block + te.size - 1])
            if te.op == 'W':
                td.write_requests += 1
                td.writes_volume += te.size
                write_intervals.append([te.block, te.block + te.size - 1])

    td.unique_reads = sum_unique_blocks(read_intervals)
    td.unique_writes = sum_unique_blocks(write_intervals)
    td.unique_total = sum_unique_blocks(read_intervals + write_intervals)

    return td


def sum_unique_blocks(intervals: List[List]) -> int:
    sum = 0
    if not intervals:
        return sum
    
    intervals.sort(key=operator.itemgetter(0))

    united = intervals[0]
    # 'united' is an interval that we will use to unite intervals.
    #  It will contain the union of the intervals that can be combined.
    for interval in intervals[1:]:
        if united[1] >= interval[0]:
            # 'united' and 'interval' can be combined.
            united[1] = max(united[1], interval[1])
        else:
            # Count the blocks of united into 'sum',
            #  and 'interval' will be the next 'united'.
            sum += united[1] - united[0] + 1
            united = interval
    sum += united[1] - united[0] + 1

    return sum


# ======= Processing the traces into 'unique_blocks.csv' =======
def write_unique_blocks_csv(traces: List[TraceData], output_path: str) -> None:

    with open(output_path, 'w') as unique_blocks_fd:
        writer = csv.writer(unique_blocks_fd)

        # titles for each column
        writer.writerow(["trace", "total unique", "unique read", 
                         "unique read ratio", "unique written", "unique written ratio"])

        for t in traces:
            total_requests = t.read_requests + t.write_requests
            unique_read_ratio = 100 * t.read_requests / total_requests
            unique_written_ratio = 100 * t.write_requests / total_requests
            writer.writerow([t.name, 
                             t.unique_total, 
                             t.unique_reads, 
                             "%.1f" % unique_read_ratio, 
                             t.unique_writes, 
                             "%.1f" % unique_written_ratio])


# ======= Processing the traces into 'accessed_blocks.csv' =======
def write_accessed_blocks_csv(traces: List[TraceData], output_path: str) -> None:

    with open(output_path, 'w') as accessed_blocks_fd:
        writer = csv.writer(accessed_blocks_fd)

        # titles for each column
        writer.writerow(["trace", "total accessed", "total read", 
                         "total read ratio", "total written", "total written ratio"])

        for t in traces:
            total_accessed = t.reads_volume + t.writes_volume
            total_read_ratio = 100 * t.unique_reads / t.unique_total
            total_written_ratio = 100 * t.unique_writes / t.unique_total
            writer.writerow([t.name, 
                            total_accessed,
                            t.reads_volume, 
                            "%.1f" % total_read_ratio,
                            t.writes_volume, 
                            "%.1f" % total_written_ratio])


def create_unique_blocks_figure(traces: List[TraceData], output_path: str):

    _, axes = plt.subplots()

    traces_name = [t.name for t in traces]
    unique_write = [t.unique_writes for t in traces]
    unique_read = [t.unique_reads for t in traces]
    unique_total = [t.unique_total for t in traces]
    
    indices = np.arange(len(traces_name))
    width = 0.35

    axes.bar(indices - width/2, unique_write, width, label='Unique Write Blocks', color='green')
    axes.bar(indices - width/2, unique_read, width, bottom=unique_write, label='Unique Read Blocks', color='orange')
    axes.bar(indices + width/2, unique_total, width, label='Total Unique Blocks', color='gray')

    axes.set_xticks(indices)
    axes.set_xticklabels(traces_name)
    # Custom y-ticks
    max_value = max(unique_total)
    step = max_value / 5
    yticks = np.arange(0, max_value + step, step)
    axes.set_yticks(yticks)
    axes.set_yticklabels([f'{int(tick/1000000):,}' for tick in yticks])

    axes.set_title('Unique Blocks')
    axes.set_xlabel('TRACES NAME')
    axes.set_ylabel('BLOCKS (in millions)')
    axes.legend()
    
    plt.gcf().savefig(output_path)


def create_accessed_blocks_figure(traces: List[TraceData], output_path: str):

    _, axes = plt.subplots()

    traces_name = [t.name for t in traces]
    total_write = [t.writes_volume for t in traces]
    total_read = [t.reads_volume for t in traces]
    
    indices = np.arange(len(traces_name))
    width = 0.5

    axes.bar(indices, total_write, width, label='Total Write Blocks', color='green')
    axes.bar(indices, total_read, width, bottom=total_write, label='Total Read Blocks', color='orange')

    axes.set_xticks(indices)
    axes.set_xticklabels(traces_name)
    # Custom y-ticks
    max_value = max(total_write) + max(total_read)
    step = max_value / 5
    yticks = np.arange(0, max_value + step, step)
    axes.set_yticks(yticks)
    axes.set_yticklabels([f'{int(tick/1000000):,}' for tick in yticks])

    axes.set_title('Accessed Blocks')
    axes.set_ylabel('BLOCKS (in millions)')
    axes.set_xlabel('TRACES NAME')
    axes.legend()
    
    plt.gcf().savefig(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_traces", nargs='+', type=str, help="path to input traces files. One or more files.")
    parser.add_argument("--out", type=str, default=os.getcwd(),
                        help="path to output directory. The code will generate output.csv and output.png")

    args = parser.parse_args()

    traces_data = []
    for trace in args.input_traces:
        result = calculate_trace_data(trace)
        traces_data.append(result)

    write_unique_blocks_csv(traces_data, os.path.join(args.out, "unique_blocks.csv"))
    write_accessed_blocks_csv(traces_data, os.path.join(args.out, "accessed_blocks.csv"))

    create_unique_blocks_figure(traces_data, os.path.join(args.out, "unique_blocks_hist.png"))
    create_accessed_blocks_figure(traces_data, os.path.join(args.out, "accessed_blocks_hist.png"))