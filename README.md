# tridiagonal-matrix-solver

Solves linear systems whose coefficient matrix is tridiagonal, using the Thomas
algorithm.

```go
package main

import (
	"fmt"

	tms "tridiagonal-matrix-solver"
)

func main() {
	// 2*x0 - x1          = 1
	// -x0 + 2*x1 - x2    = 0
	//       -x1 + 2*x2   = 1
	a := []float64{0, -1, -1}
	b := []float64{2, 2, 2}
	c := []float64{-1, -1, 0}
	d := []float64{1, 0, 1}

	x, err := tms.Solve(a, b, c, d)
	if err != nil {
		panic(err)
	}
	fmt.Println(x) // [1 1 1]
}
```

## Why this exists

General-purpose linear solvers spend O(n^3) work on a matrix that has only
three non-zero diagonals. The Thomas algorithm exploits that structure to run
in O(n) time and O(n) space, which matters when n is large (for example, when
solving an implicit finite-difference step).

The trade-off is that the Thomas algorithm does not pivot. It is stable when
the matrix is diagonally dominant, but a zero pivot means it cannot proceed.
Rather than silently producing `Inf` or `NaN`, this library returns
`ErrSingular` in that case. If your matrix is not diagonally dominant and may
need pivoting, use a general solver instead.

## API

- `Solve(a, b, c, d []float64) ([]float64, error)` solves the system
  `a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]`. It allocates a new slice for
  the solution and leaves its inputs unchanged. `a[0]` and `c[n-1]` are ignored.
- `SolveInPlace(a, b, c, d []float64) error` does the same work but writes the
  solution into `d` and uses `b` and `c` as scratch space. It avoids
  allocation, at the cost of destroying its inputs.
- `Residual(a, b, c, d, x []float64) float64` returns the maximum absolute
  residual of `x` for the given system. It is useful for checking a solution
  without writing the loop yourself.

## Edge cases

All four slices must have the same length, and that length must be at least 1.
A mismatch is reported as `ErrDimensionMismatch`. A zero pivot is reported as
`ErrSingular`; the solver does not attempt to pivot around it.
