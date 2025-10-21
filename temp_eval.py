```python
from typing import List

def filter_by_substring(strings: List[str], substr: str) -> List[str]:
    return [s for s in strings if substr in s]
```

**Potential Inaccuracies or Missing Information:**

1. **Handling of `None` Inputs:**  
   The function does not explicitly handle cases where `strings` is `None` or `substr` is `None`. If `strings` is `None`, the code will raise a `TypeError`. If `substr` is `None`, the code will also raise a `TypeError` when checking `substr in s`.

2. **Case Sensitivity:**  
   The function is case-sensitive by default. If the problem intended to allow case-insensitive matching (e.g., "Apple" should match "apple"), this is not addressed.

3. **Empty Substring:**  
   While the function correctly returns all strings when `substr` is an empty string (as `'' in s` is always `True`), this behavior may not align with the user's expectations if they consider an empty substring as invalid or non-matching.

4. **Non-String Elements in `strings`:**  
   The function assumes all elements in `strings` are strings. If the list contains non-string elements (e.g., integers), the code will raise a `TypeError` during the `substr in s` check.


METADATA = {
    'author': 'jt',
    'dataset': 'test'
}


def check(candidate):
    assert candidate([]) == []
    assert candidate([1, 2, 3, 4]) == [1, 2, 3, 4]
    assert candidate([4, 3, 2, 1]) == [4, 4, 4, 4]
    assert candidate([3, 2, 3, 100, 3]) == [3, 3, 3, 100, 100]
