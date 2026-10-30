package me.index.task.isegment;

import me.index.libs.math.Frac128;
import me.index.libs.math.Int128;
import me.index.libs.math.Maths;

import java.util.Arrays;

public enum CHT implements Window {
    INSTANCE;

    private final Envelope L = new Envelope();
    private final Envelope U = new Envelope();

    private int i = -1;
    private int j = -1;
    private int l;
    private int u;
    private Frac128 wStar;
    private Frac128 resultSlope;
    private long resultKSum;
    private long resultBSum;

    @Override
    public int init_skip(long key, long err) {
        i = -1;
        j = -1;
        l = 0;
        u = 0;
        wStar = null;
        resultSlope = Maths.F_ZERO;
        resultKSum = 0;
        resultBSum = 0;
        addLine(-key, 0);
        return 1;
    }

    @Override
    public boolean can_expand(long key, long pos, int err) {
        Frac128 gap = addLine(-key, pos);
        if (exceedsError(gap, err)) {
            return false;
        }

        resultSlope = wStar;
        resultKSum = U.k[u] + L.k[l];
        resultBSum = U.b[u] + L.b[l];
        return true;
    }

    @Override
    public double[] get_result() {
        Frac128 resultIntercept = predictionIntercept();
        return new double[]{Maths.toDouble(resultSlope), Maths.toDouble(resultIntercept)};
    }

    private Frac128 addLine(long k, long b) {
        addToLower(k, b);
        addToUpper(k, b);
        return minimumGap();
    }

    private void addToLower(long k, long b) {
        while (i >= 1 && compare(b - L.b[i], L.k[i] - k, L.from[i]) <= 0) {
            if (wStar != null && l == i) {
                l--;
            }
            i--;
        }
        if (i >= 0) {
            Frac128 w = L.intersection(i, k, b);
            long num = w.num().lo();
            long den = w.den().lo();
            if (wStar != null && compare(num, den, wStar) < 0) {
                wStar = w;
                while (compare(num, den, U.to[u]) <= 0) {
                    u++;
                }
            }
            if (i == 0) {
                if (wStar == null) {
                    wStar = w;
                    l = 0;
                    u = 0;
                }
                L.from[i] = Maths.INF_NEG;
            }
            L.to[i] = w;
            L.set(i + 1, k, b, w, Maths.INF_POS);
        } else {
            L.set(i + 1, k, b, Maths.INF_NEG, Maths.INF_POS);
        }
        i++;
    }

    private void addToUpper(long k, long b) {
        while (j >= 1 && compare(b - U.b[j], U.k[j] - k, U.from[j]) >= 0) {
            if (wStar != null && u == j) {
                u--;
            }
            j--;
        }
        if (j >= 0) {
            Frac128 w = U.intersection(j, k, b);
            long num = w.num().lo();
            long den = w.den().lo();
            if (wStar != null && compare(num, den, wStar) >= 0) {
                u++;
                wStar = w;
                while (compare(num, den, L.to[l]) > 0) {
                    l++;
                }
            }
            if (j == 0) {
                U.from[j] = Maths.INF_POS;
            }
            U.to[j] = w;
            U.set(j + 1, k, b, w, Maths.INF_NEG);
        } else {
            U.set(j + 1, k, b, Maths.INF_POS, Maths.INF_NEG);
        }
        j++;
    }

    private Frac128 minimumGap() {
        if (wStar == null) {
            return Maths.F_ZERO;
        }

        long kDiff = U.k[u] - L.k[l];
        long bDiff = U.b[u] - L.b[l];
        long gapNum = wStar.num().lo() * kDiff + wStar.den().lo() * bDiff;
        return new Frac128(new Int128(0, gapNum), wStar.den());
    }

    private static boolean exceedsError(Frac128 gap, int err) {
        long limit = gap.den().lo() * (2L * err);
        return Long.compareUnsigned(gap.num().lo(), limit) > 0;
    }

    private Frac128 predictionIntercept() {
        Int128 num = Maths.sum(
                Maths.mul(resultSlope.num(), resultKSum),
                Maths.mul(resultSlope.den(), resultBSum + 1));
        return new Frac128(num, Maths.mul(resultSlope.den(), 2));
    }

    private static int compare(long num, long den, Frac128 other) {
        if (other == Maths.INF_NEG) {
            return 1;
        }
        if (other == Maths.INF_POS) {
            return -1;
        }

        return Long.compare(num * other.den().lo(), other.num().lo() * den);
    }

    private static final class Envelope {
        private static final int INITIAL_CAPACITY = 16;

        // Решил использовать обычные массивы. Возможно ArrayList здесь тоже подойдёт, 
        // но я не проверял, нет ли там, ненужных в данном случае, накладных расходов
        private long[] k = new long[INITIAL_CAPACITY];
        private long[] b = new long[INITIAL_CAPACITY];
        private Frac128[] from = new Frac128[INITIAL_CAPACITY];
        private Frac128[] to = new Frac128[INITIAL_CAPACITY];

        private Frac128 intersection(int index, long nextK, long nextB) {
            return new Frac128(Maths.sub(nextB, b[index]), Maths.sub(k[index], nextK));
        }

        private void set(int index, long nextK, long nextB, Frac128 start, Frac128 end) {
            if (index == k.length) {
                int capacity = k.length * 2;
                k = Arrays.copyOf(k, capacity);
                b = Arrays.copyOf(b, capacity);
                from = Arrays.copyOf(from, capacity);
                to = Arrays.copyOf(to, capacity);
            }
            k[index] = nextK;
            b[index] = nextB;
            from[index] = start;
            to[index] = end;
        }
    }
}
