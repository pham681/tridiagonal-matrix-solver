// Package tridiagonal_matrix_solver solves linear systems with tridiagonal
// coefficient matrices using the Thomas algorithm.
//
// A tridiagonal matrix has non-zero entries only on the main diagonal and the
// two adjacent diagonals. Such systems arise from finite-difference
// discretisations of second-order ODEs, cubic spline interpolation, and
// implicit time stepping. The Thomas algorithm solves them in O(n) time and
// O(n) space, which is much cheaper than general Gaussian elimination.
//
// The solver does not pivot. If a diagonal pivot is zero (or very close to
// zero) the system is reported as singular rather than producing Inf/NaN.
package tridiagonal_matrix_solver

import (
	"errors"
	"math"
)

// ErrSingular is returned when a pivot is zero or effectively zero, so no
// unique solution can be computed without pivoting.
var ErrSingular = errors.New("tridiagonal_matrix_solver: singular matrix (zero pivot)")

// ErrDimensionMismatch is returned when the slices describing the system do
// not all have the same length, or when n is zero.
var ErrDimensionMismatch = errors.New("tridiagonal_matrix_solver: dimension mismatch")

// Solve solves the tridiagonal system
//
//	a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]   for i = 0, ..., n-1
//
// with a[0] and c[n-1] ignored (there is no x[-1] or x[n]). The three diagonal
// slices and the right-hand side must all have the same length n >= 1.
//
// The returned slice is freshly allocated; the inputs are not modified.
func Solve(a, b, c, d []float64) ([]float64, error) {
	n := len(b)
	if n == 0 {
		return nil, ErrDimensionMismatch
	}
	if len(a) != n || len(c) != n || len(d) != n {
		return nil, ErrDimensionMismatch
	}

	// Work on copies so the caller's slices are left untouched. cp and dp are
	// the modified upper diagonal and right-hand side produced by forward
	// elimination; x holds the solution.
	cp := make([]float64, n)
	dp := make([]float64, n)
	x := make([]float64, n)

	// The first pivot is just b[0]. A zero here means the matrix is singular
	// (without pivoting) and we refuse to divide by it.
	if b[0] == 0 {
		return nil, ErrSingular
	}
	cp[0] = c[0] / b[0]
	dp[0] = d[0] / b[0]

	// Forward sweep: eliminate a[i] using the previous row.
	for i := 1; i < n; i++ {
		denom := b[i] - a[i]*cp[i-1]
		if denom == 0 {
			return nil, ErrSingular
		}
		cp[i] = c[i] / denom
		dp[i] = (d[i] - a[i]*dp[i-1]) / denom
	}

	// Back substitution. The last unknown is already isolated.
	x[n-1] = dp[n-1]
	for i := n - 2; i >= 0; i-- {
		x[i] = dp[i] - cp[i]*x[i+1]
	}

	return x, nil
}

// SolveInPlace is like Solve but writes the solution into d and uses b as
// scratch space. It is provided for callers that already own their slices and
// want to avoid allocating. On any error the contents of b and d are
// unspecified.
func SolveInPlace(a, b, c, d []float64) error {
	n := len(b)
	if n == 0 {
		return ErrDimensionMismatch
	}
	if len(a) != n || len(c) != n || len(d) != n {
		return ErrDimensionMismatch
	}

	if b[0] == 0 {
		return ErrSingular
	}
	c[0] /= b[0]
	d[0] /= b[0]

	for i := 1; i < n; i++ {
		denom := b[i] - a[i]*c[i-1]
		if denom == 0 {
			return ErrSingular
		}
		c[i] /= denom
		d[i] = (d[i] - a[i]*d[i-1]) / denom
	}

	for i := n - 2; i >= 0; i-- {
		d[i] -= c[i] * d[i+1]
	}
	return nil
}

// Residual returns the maximum absolute residual max_i |a[i]*x[i-1] + b[i]*x[i]
// + c[i]*x[i+1] - d[i]| for the given system and candidate solution. It is a
// convenience for callers that want to verify a solution without writing the
// loop themselves. The boundary terms involving a[0] and c[n-1] are ignored.
func Residual(a, b, c, d, x []float64) float64 {
	n := len(b)
	if len(a) != n || len(c) != n || len(d) != n || len(x) != n || n == 0 {
		return math.Inf(1)
	}
	var max float64
	for i := 0; i < n; i++ {
		var sum float64
		sum += b[i] * x[i]
		if i > 0 {
			sum += a[i] * x[i-1]
		}
		if i < n-1 {
			sum += c[i] * x[i+1]
		}
		if r := math.Abs(sum - d[i]); r > max {
			max = r
		}
	}
	return max
}
