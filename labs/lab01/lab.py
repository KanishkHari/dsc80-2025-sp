# lab.py


from pathlib import Path
import io
import pandas as pd
import numpy as np
np.set_printoptions(legacy='1.21')


# ---------------------------------------------------------------------
# QUESTION 0
# ---------------------------------------------------------------------


def consecutive_ints(ints):
    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def median_vs_mean(nums):
  result = False
  # mean
  mean = sum(nums) / len(nums)
  # sort numbers to find median
  nums_sorted = sorted(nums)
  length = len(nums_sorted)
  # calculating median
  # if length is odd
  if length % 2 == 1:
    median = nums_sorted[int(length / 2)]
  # if length is even
  else:
    median = (nums_sorted[int(length / 2 - 1)] + nums_sorted[int(length/2)]) / 2
    
  if median <= mean:
    result = True
  return result
    


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def n_prefixes(s, n):
  prefix = []
  for i in range(1, n+1):
    prefix.append(s[:i])
  reverse = prefix[::-1]
  result = ''.join(reverse)
  return result


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def exploded_numbers(ints, n):
    answer = []
    max_num = max(i + n for i in ints)
    digits = str(max_num)
    len_digits = len(digits)
    for i in ints:
        exploded_range = range(i-n, i+n+1)
        padded_numbers = [str(j).zfill(len_digits) for j in exploded_range]
        answer.append(' '.join(padded_numbers))
    return answer
    


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def last_chars(fh):
  result = []
  for line in fh:
    stripped = line.rstrip('\n')
    if stripped:
      result.append(stripped[-1])
  return ''.join(result)



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def add_root(A):
    square_root = np.sqrt(np.arange(len(A)))
    return A + square_root

def where_square(A):
    square_root = np.sqrt(A)
    return np.isclose(square_root, np.round(square_root))  


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def filter_cutoff_loop(matrix, cutoff):
# get row and col number
    num_rows = len(matrix)
    if num_rows == 0:
        return []
    num_cols = len(matrix[0])
    # storing indexes of columns
    columns_kept = []

    for column_index in range(num_cols):
        column_sum = sum(matrix[row][column_index] for row in range(num_rows))
        mean = column_sum / num_rows
        if mean > cutoff:
            columns_kept.append(column_index)

    filtered_matrix = [[matrix[row][col] for col in columns_kept] for row in range(num_rows)]
    
    return np.array(filtered_matrix)  # Convert the result to a numpy array
   


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def filter_cutoff_np(matrix, cutoff):
    mean = np.mean(matrix, axis=0)
    return matrix[:, mean > cutoff]


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def growth_rates(A):
  # declare variable rates that returns difference between A and last element of A
  rates = np.diff(A) / A[:-1]
  # return np,array of rates to 2 decimal places
  return np.round(rates, 2)

def with_leftover(A):
    # declare variable leftover as 20 % A
    leftover = 20 % A
    # cumulative leftover is based on added leftover from previous days
    leftover_cumulative = np.cumsum(leftover)
    # first day where cumulative leftover is greater than A
    cond = leftover_cumulative >= A
    if np.any(cond):
        return np.argmax(cond)
    else:
        return -1


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def salary_stats(salary):
  # num players
  num_players = salary.shape[0]
  # num of teams
  num_teams = salary['Team'].unique()
  # total salary
  total_salary = salary['Salary'].sum()
  # highest salary
  highest_salary = salary.sort_values('Salary', ascending = False)['Player'].iloc[0]
  # average los angeles laker salary
  avg_los = round(salary[salary['Team'] == 'Los Angeles Lakers']['Salary'].mean(), 2)
  # fifth lowest salary in league
  fifth_lowest = salary.sort_values('Salary').iloc[4, 0] + ', ' + salary.sort_values('Salary').iloc[4, 2]
  # duplicates -> 1. create variable that strips last name; 2. use that variable in duplicate
  last_name = salary['Player'].str.split(' ').str[-1]
  duplicates = last_name.duplicated().any()
  # highest collective salary of all teams in league
  total_highest = salary[salary['Team'] == salary.sort_values('Salary', ascending = False)['Team'].iloc[0]]['Salary'].sum()
  result = pd.Series([num_players, num_teams, total_salary, highest_salary, avg_los, fifth_lowest, duplicates, total_highest], index = ['num_players', 'num_teams', 'total_salary', 'highest_salary', 'avg_los', 'fifth_lowest', 'duplicates', 'total_highest'])
  return result



# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def parse_malformed(fp):
   def parse_malformed(fp):
    columns = ["first", "last", "weight", "height", "x-geo", "y-geo"]
    rows = []
    df = pd.DataFrame(rows, columns=columns)

    with open(fp) as file:
        i = 0
        for row in file:
            if i != 0:
                new_row = row.replace('"', '').replace(',', ' ').split()
                df.loc[len(df)] = new_row
            else:
                i += 1

 
    df["height"] = df["height"].astype(float)
    df["weight"] = df["weight"].astype(float)

    df["geo"] = df["x-geo"] + "," + df["y-geo"]

    df = df.drop(columns=["x-geo", "y-geo"])

    return df
