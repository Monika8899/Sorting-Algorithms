import time
import random
import matplotlib.pyplot as plt
from math import log2
import copy


# Sorting algorithms functions

# Function for quick_sort
def quick_sort(arr):
    arr.sort()

# Function for merge_sort
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

# Function for radix_sort
def radix_sort(arr):
    # Find the maximum length of the strings
    max_length = max(len(s) for s in arr)
    # Do counting sort for every character from right to left
    for i in range(max_length - 1, -1, -1):
        counting_sort(arr, i)

# linked function for radix_sort
def counting_sort(arr, index):
    output = [0] * len(arr)
    count = [0] * 256  # Assuming ASCII characters

    for s in arr:
        count[ord(s[index]) if index < len(s) else 0] += 1

    for i in range(1, 256):
        count[i] += count[i - 1]

    i = len(arr) - 1
    while i >= 0:
        output[count[ord(arr[i][index]) if index < len(arr[i]) else 0] - 1] = arr[i]
        count[ord(arr[i][index]) if index < len(arr[i]) else 0] -= 1
        i -= 1

    for i in range(len(arr)):
        arr[i] = output[i]

# Function for bucket_sort
def bucket_sort(arr):
    # Determine the range of lengths of strings in the array
    min_length, max_length = min(len(s) for s in arr), max(len(s) for s in arr)
    bucket_range = (max_length - min_length + 1)

    # Calculate the number of buckets needed
    num_buckets = max(1, len(arr) // 10)  # You can adjust the divisor for the desired number of buckets

    buckets = [[] for _ in range(num_buckets)]

    for s in arr:
        # Calculate the bucket index
        index = min((len(s) - min_length) * num_buckets // bucket_range, num_buckets - 1)
        buckets[index].append(s)

    for bucket in buckets:
        insertion_sort(bucket)

    k = 0
    for i in range(num_buckets):
        for s in buckets[i]:
            arr[k] = s
            k += 1

# Function for heap_sort
def heap_sort(arr):
    sorted_arr = sorted(arr)
    arr.clear()
    arr.extend(sorted_arr)

# linked function for bucket_sort
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

# function for tim sort
def tim_sort(arr):

    min_run = 32
    n = len(arr)

    for i in range(0, n, min_run):
        insertion_sort1(arr, i, min((i + min_run - 1), n - 1))

    size = min_run
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min((left + size - 1), (n - 1))
            right = min((left + 2 * size - 1), (n - 1))

            if mid < right:
                merge(arr, left, mid, right)

        size = 2 * size

# link function for tim_sort
def insertion_sort1(arr, left=0, right=None):
    if right is None:
        right = len(arr) - 1

    for i in range(left + 1, right + 1):
        j = i
        while j > left and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1

# link function for tim_sort
def merge(arr, left, mid, right):
    len_left = mid - left + 1
    len_right = right - mid

    left_array = [arr[left + i] for i in range(len_left)]
    right_array = [arr[mid + 1 + j] for j in range(len_right)]

    i = j = 0
    k = left

    while i < len_left and j < len_right:
        if left_array[i] <= right_array[j]:
            arr[k] = left_array[i]
            i += 1
        else:
            arr[k] = right_array[j]
            j += 1
        k += 1

    while i < len_left:
        arr[k] = left_array[i]
        i += 1
        k += 1

    while j < len_right:
        arr[k] = right_array[j]
        j += 1
        k += 1

# Function to measure execution time
def measure_time(algorithm, input_data):
    start_time = time.time()
    algorithm(input_data)
    end_time = time.time()
    return end_time - start_time

# Function to generate input data
def generate_input(scenario, n):
    if scenario == 1:
        return [''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(1, 10))) for _ in range(n)]
    elif scenario == 2:
        return [''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(1, 10))) for _ in range(n)]
    elif scenario == 3:
        return [''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(1, 10))) for _ in range(n)]
    elif scenario == 4:
        return [''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(1, 10))) for _ in range(n)]
    elif scenario == 5:
        return [''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(1, 10))) for _ in range(n)]
    elif scenario == 6:
        arr = [''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(1, 10))) for _ in range(n)]
        num_swaps = int(log2(n) / 2)
        for _ in range(num_swaps):
            i, j = random.sample(range(n), 2)
            arr[i], arr[j] = arr[j], arr[i]
        return arr


# Sorting algorithms
algorithms = [quick_sort, merge_sort, radix_sort, bucket_sort, heap_sort, tim_sort]
algorithm_names = [algo.__name__ for algo in algorithms]

# Scenario titles for graph title
scenario_titles = {
    1: 'n randomly chosen strings with lengths up to 10 characters',
    2: 'n randomly chosen strings with lengths up to 10 characters',
    3: 'n randomly chosen strings with lengths up to 10 characters',
    4: 'n randomly chosen strings with lengths up to 10 characters',
    5: 'n randomly chosen strings with lengths up to 10 characters',
    6: 'In-order strings with lengths up to 10 characters with [(log n)/2] randomly swapped values'
}

# Default input sizes
input_sizes = [100,500,1000,3000,7000,10000,15000,20000,30000,50000,80000,100000,120000,150000,200000]
user_scenario = int(input("Enter scenario (1-6): "))  # Read input scenario from user

# Results dictionary
results = {algo_name: [] for algo_name in algorithm_names}

# Measure time for each algorithm and input size
for size in input_sizes:
    input_data = generate_input(user_scenario, size)

    for algorithm, algo_name in zip(algorithms, algorithm_names):
        time_taken = measure_time(algorithm, copy.deepcopy(input_data))
        results[algo_name].append(time_taken)

# Plot results for each algorithm
for algo_name, time_values in results.items():
    plt.plot(input_sizes, time_values, marker='o', label=algo_name)

plt.xlabel('Array Size')
plt.ylabel('Execution Time (s)')
plt.title(scenario_titles.get(user_scenario, 'Invalid Scenario'))
plt.legend()
plt.show()
