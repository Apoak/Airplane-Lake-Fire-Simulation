# Download all "href" entities from a file

import argparse
import os
import re
import requests
import shutil
import signal
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

from interrupt import handle_ctrl_c, init_pool

def run_command(command_string, verbose=False, suppress_output=False):
    if verbose:
        print("> " + command_string)

    kwargs = {
        "shell": True,
    }
    if suppress_output:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
        
    retval = subprocess.call(command_string, **kwargs)
    return retval

def maybe_create_directory(dir_name):
    if os.path.isdir(dir_name):
        return
    os.mkdir(dir_name)
    if not os.path.isdir(dir_name):
        print(f"Couldn't create directory {dir_name}")

@handle_ctrl_c
def process_tile(args):
    (directory, url) = args

    # Get filename section
    match = re.search(r"/([^/]*?)$", url)
    filename = os.path.join(directory, match.group(1))
    
    # Skip file if it already exists
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"Skipping {filename}")
        return
    
    print(f"Downloading {filename}")

    # Download to a temp location and move, to avoid partial downloads
    temp_dir = tempfile.TemporaryDirectory()
    temp_filename = os.path.join(temp_dir.name, "downloading")
    
    while True:
        # Downloading from slow sites manually tended to fail; just use curl
        command = f"curl -s --retry-all-errors -o {temp_filename} {url}"
        retval = run_command(command, verbose=False)
        if retval == 0 and Path(temp_filename).exists():
            break
        print(f"Retrying {filename}")

    # Move file to final location
    shutil.move(temp_filename, filename)
    temp_dir.cleanup()
        
def main():
    parser = argparse.ArgumentParser(description='Download all HREFs from files to directory')
    parser.add_argument('inputs', nargs='+', help='Input files')
    parser.add_argument('--output_directories', default = '.',
                        help="Directories to contain output files, comma-separated")
    parser.add_argument('--match_regexp', default = '.*',
                        help="Regular expression to constrain downloaded filenames")
    parser.add_argument('--threads', default=1, type=int,
                        help="Number of parallel jobs to run")
    args = parser.parse_args()

    output_dirs = args.output_directories.split(',')
    for output_dir in output_dirs:
        maybe_create_directory(output_dir)

    # Run in parallel over each peak inside the shapefile
    original_signal_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = Pool(args.threads, initializer=init_pool)
    signal.signal(signal.SIGINT, original_signal_handler)
    process_args = []

    for index, each_input in enumerate(args.inputs):
        # Where should the files for this input go?
        output_dir = output_dirs[min(index, len(output_dirs) - 1)]
        
        # Download input file?
        if each_input.startswith("http"):
            r = requests.get(each_input)
            contents = r.text
        else:
            with open(each_input, "r") as f:
                contents = f.read()

        # Find all links in file
        href_pattern1 = r'(https://[^<\n"]*)'
        href_pattern2 = r'href="(.*?)"'
        matches1 = re.findall(href_pattern1, contents) or []
        matches2 = re.findall(href_pattern2, contents) or []
        matches1.extend(matches2)
        matches = list(set(matches1))

        file_pattern = re.compile(args.match_regexp)
        for url in matches:
            # Build absolute URL if necessary
            if not file_pattern.match(url):
                url = each_input + url
            
            if file_pattern.match(url):
                # Convert relative URL to absolute?
                if each_input.startswith("http"):
                    url = requests.compat.urljoin(each_input, url)
                process_args.append((output_dir, url))

    results = pool.map_async(process_tile, process_args).get(999999)
    if any(map(lambda x: isinstance(x, KeyboardInterrupt), results)):
        print('Ctrl-C was entered.')
        exit(1)

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
