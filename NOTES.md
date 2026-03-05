# Notes

## 2026-03-05

### microgpt.cpp Changes
- Removed `operator[]` from Vector class, now uses explicit `.data` access for clarity
- Changed `vec[j]` to `vec.data[j]` in Matrix::operator*
- Changed all `a[i]`, `v[i]` to `a.data[i]`, `v.data[i]` in add/scale functions
- This makes the code more explicit about what it's doing (accessing internal vector)
