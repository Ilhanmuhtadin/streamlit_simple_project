def maxposs(A, low, high):
    if low == high:
        return low
    else:
        print(low)
        posti = maxposs(A, low + 1, high)
        if A[low] > A[posti]:
            posti = low
        return posti

# Example usage:
# Replace `your_array` with the actual array you want to find the maximum index in.
your_array = [4, 2, 7, 1, 9, 5]
result = maxposs(your_array, 0, len(your_array) - 1)
print(f"The index of the maximum element is: {result}")
