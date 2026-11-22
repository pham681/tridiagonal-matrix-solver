package tridiagonal_matrix_solver

import (
	"errors"
	"math"
	"testing"
)

const eps = 1e-9

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < eps
}

func TestSolveSimpleSystem(t *testing.T) {
	// 2x0 - x1 = 1
	// -x0 + 2x1 - x2 = 0
	// -x1 + 2x2 = 1
	// The exact solution is x = (1, 1, 1).
	a := []float64{0, -1, -1}
	b := []float64{2, 2, 2}
	c := []float64{-1, -1, 0}
	d := []float64{1, 0, 1}

	x, err := Solve(a, b, c, d)
	if err != nil {
		t.Fatalf("Solve returned error: %v", err)
	}
	want := []float64{1, 1, 1}
	for i := range want {
		if !almostEqual(x[i], want[i]) {
			t.Errorf("x[%d] = %v, want %v", i, x[i], want[i])
		}
	}
}

func TestSolveSingleElement(t *testing.T) {
	// A 1x1 system: b[0]*x[0] = d[0].
	x, err := Solve([]float64{0}, []float64{4}, []float64{0}, []float64{8})
	if err != nil {
		t.Fatalf("Solve returned error: %v", err)
	}
	if len(x) != 1 || !almostEqual(x[0], 2) {
		t.Fatalf("x = %v, want [2]", x)
	}
}

func TestSolveDoesNotModifyInputs(t *testing.T) {
	a := []float64{0, 1, 1}
	b := []float64{3, 3, 3}
	c := []float64{1, 1, 0}
	d := []float64{4, 6, 4}

	// Copy the inputs so we can compare afterwards.
	aCopy := append([]float64(nil), a...)
	bCopy := append([]float64(nil), b...)
	cCopy := append([]float64(nil), c...)
	dCopy := append([]float64(nil), d...)

	if _, err := Solve(a, b, c, d); err != nil {
		t.Fatalf("Solve returned error: %v", err)
	}

	for i := range a {
		if a[i] != aCopy[i] || b[i] != bCopy[i] || c[i] != cCopy[i] || d[i] != dCopy[i] {
			t.Fatalf("Solve modified its inputs at index %d", i)
		}
	}
}

func TestSolveDimensionMismatch(t *testing.T) {
	cases := []struct {
		name string
		a, b, c, d []float64
	}{
		{"empty", []float64{}, []float64{}, []float64{}, []float64{}},
		{"b longer than a", []float64{0}, []float64{1, 2}, []float64{0, 0}, []float64{1, 2}},
		{"b longer than c", []float64{0, 0}, []float64{1, 2}, []float64{0}, []float64{1, 2}},
		{"b longer than d", []float64{0, 0}, []float64{1, 2}, []float64{0, 0}, []float64{1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Solve(tc.a, tc.b, tc.c, tc.d)
			if !errors.Is(err, ErrDimensionMismatch) {
				t.Fatalf("err = %v, want ErrDimensionMismatch", err)
			}
		})
	}
}

func TestSolveSingular(t *testing.T) {
	// b[0] = 0 makes the first pivot zero.
	_, err := Solve([]float64{0}, []float64{0}, []float64{0}, []float64{1})
	if !errors.Is(err, ErrSingular) {
		t.Fatalf("err = %v, want ErrSingular", err)
	}

	// A later pivot becomes zero: row 1 has b[1] - a[1]*cp[0] = 2 - 1*1 = 1,
	// row 2 has b[2] - a[2]*cp[1] = 0 - 1*1 = -1, which is fine. Use a case
	// where the second pivot is exactly zero: b = [1, 1, 1], a = [0, 1, 0],
	// c = [1, 0, 0]. Then cp[0] = 1, denom = 1 - 1*1 = 0.
	_, err = Solve([]float64{0, 1, 0}, []float64{1, 1, 1}, []float64{1, 0, 0}, []float64{1, 1, 1})
	if !errors.Is(err, ErrSingular) {
		t.Fatalf("err = %v, want ErrSingular", err)
	}
}

func TestSolveInPlaceMatchesSolve(t *testing.T) {
	a := []float64{0, -1, -1}
	b := []float64{2, 2, 2}
	c := []float64{-1, -1, 0}
	d := []float64{1, 0, 1}

	x, err := Solve(a, b, c, d)
	if err != nil {
		t.Fatalf("Solve returned error: %v", err)
	}

	a2 := append([]float64(nil), a...)
	b2 := append([]float64(nil), b...)
	c2 := append([]float64(nil), c...)
	d2 := append([]float64(nil), d...)

	if err := SolveInPlace(a2, b2, c2, d2); err != nil {
		t.Fatalf("SolveInPlace returned error: %v", err)
	}

	for i := range x {
		if !almostEqual(x[i], d2[i]) {
			t.Errorf("SolveInPlace result[%d] = %v, want %v", i, d2[i], x[i])
		}
	}
}

func TestSolveInPlaceErrors(t *testing.T) {
	if err := SolveInPlace([]float64{}, []float64{}, []float64{}, []float64{}); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("empty: err = %v, want ErrDimensionMismatch", err)
	}
	if err := SolveInPlace([]float64{0}, []float64{0}, []float64{0}, []float64{1}); !errors.Is(err, ErrSingular) {
		t.Fatalf("singular: err = %v, want ErrSingular", err)
	}
}

func TestResidual(t *testing.T) {
	a := []float64{0, -1, -1}
	b := []float64{2, 2, 2}
	c := []float64{-1, -1, 0}
	d := []float64{1, 0, 1}

	x, err := Solve(a, b, c, d)
	if err != nil {
		t.Fatalf("Solve returned error: %v", err)
	}
	if r := Residual(a, b, c, d, x); r > eps {
		t.Fatalf("Residual = %v, want <= %v", r, eps)
	}

	// A deliberately wrong solution should have a noticeable residual.
	bad := []float64{0, 0, 0}
	if r := Residual(a, b, c, d, bad); r < 0.5 {
		t.Fatalf("Residual of wrong solution = %v, want >= 0.5", r)
	}
}

func TestResidualDimensionMismatch(t *testing.T) {
	if r := Residual([]float64{0}, []float64{1, 2}, []float64{0, 0}, []float64{1, 2}, []float64{1, 2}); !math.IsInf(r, 1) {
		t.Fatalf("Residual = %v, want +Inf", r)
	}
	if r := Residual([]float64{}, []float64{}, []float64{}, []float64{}, []float64{}); !math.IsInf(r, 1) {
		t.Fatalf("Residual = %v, want +Inf", r)
	}
}
