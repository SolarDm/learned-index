package me.index.task.isegment;

import me.index.libs.math.Int128;
import me.index.libs.math.Maths;
import me.index.libs.math.Frac128;

import java.util.Arrays;

public enum CHT implements Window {
    INSTANCE(100);

    CHT(int size) {
        LK = new long[size];
        LB = new long[size];
        LT = new Frac128[size];
        UK = new long[size];
        UB = new long[size];
        UT = new Frac128[size];
    }

    private int lh; // lower stack head
    private int lp; // lower stack pointer
    private long[] LK; // lower stack k-s
    private long[] LB; // lower stack b-s
    private Frac128[] LT; // lower breakpoints, increasing; last is +inf

    private int uh; // upper stack head
    private int up; // upper stack pointer
    private long[] UK; // upper stack k-s
    private long[] UB; // upper stack b-s
    private Frac128[] UT; // upper breakpoints, decreasing; last is -inf

    private Frac128 ans; // slope minimizing the gap between the envelopes
    private Frac128 cf0; // last accepted slope
    private Frac128 cf1; // last accepted intercept

    @Override
    public int init_skip(long key, long err) {
        lp = -1;
        up = -1;
        lh = -1;
        uh = -1;
        ans = null;
        cf0 = Maths.F_ZERO;
        cf1 = Maths.F_ZERO;
        return 0;
    }

    @Override
    public boolean can_expand(long key, long pos, int err) {
        if (key == Long.MIN_VALUE) {
            throw new IllegalArgumentException("CHT cannot negate Long.MIN_VALUE without overflow");
        }
        long k = -key;
        long b = pos;
        while (lh > 0 && Maths.lessEq((LT[lh] = sec(LK[lh], LB[lh], k, b)), LT[lh - 1])) {
            lh--;
        }
        if (lh == 0) {
            LT[lh] = sec(LK[lh], LB[lh], k, b);
        }
        if (ans != null) {
            if (lp > lh) {
                lp = lh;
            }
            if (lh >= 0 && Maths.less(LT[lh], ans)) {
                ans = LT[lh];
                while (Maths.greatEq(UT[up], ans)) {
                    up++;
                }
            }
        } else if (lh >= 0) {
            ans = LT[lh];
            lp = 0;
            up = 0;
        }
        ++lh;
        if (lh == LK.length) {
            growLower();
        }
        LK[lh] = k;
        LB[lh] = b;
        LT[lh] = Maths.INF_POS;
        while (uh > 0 && Maths.greatEq((UT[uh] = sec(UK[uh], UB[uh], k, b)), UT[uh - 1])) {
            uh--;
        }
        if (uh == 0) {
            UT[uh] = sec(UK[uh], UB[uh], k, b);
        }
        if (ans != null) {
            if (up > uh) {
                up = uh;
            }
            if (uh >= 0 && Maths.greatEq(UT[uh], ans)) {
                ans = UT[uh];
                up++;
                while (Maths.less(LT[lp], ans)) {
                    lp++;
                }
            }
        }
        ++uh;
        if (uh == UK.length) {
            growUpper();
        }
        UK[uh] = k;
        UB[uh] = b;
        UT[uh] = Maths.INF_NEG;
        if (ans == null) {
            return true;
        }
        Int128 x = Maths.sum(Maths.mul(ans.num(), UK[up]), Maths.mul(ans.den(), UB[up]));
        Int128 y = Maths.sum(Maths.mul(ans.num(), LK[lp]), Maths.mul(ans.den(), LB[lp]));
        if (Maths.lessOrEqZero(Maths.sub(Maths.sub(x, y), Maths.mul(ans.den(), err * 2L)))) {
            cf0 = ans;
            cf1 = new Frac128(Maths.sum(x, y), Maths.mul(ans.den(), 2L));
            return true;
        }
        return false;
    }

    @Override
    public double[] get_result() {
        return new double[]{
                Maths.toDouble(cf0),
                Maths.toDouble(cf1)
        };
    }

    private static Frac128 sec(long sk, long sb, long k, long b) {
        return new Frac128(Maths.sub(b, sb), Maths.sub(sk, k));
    }

    private void growLower() {
        int size = (int) Math.min(2L * LK.length, Integer.MAX_VALUE);
        LK = Arrays.copyOf(LK, size);
        LB = Arrays.copyOf(LB, size);
        LT = Arrays.copyOf(LT, size);
    }

    private void growUpper() {
        int size = (int) Math.min(2L * UK.length, Integer.MAX_VALUE);
        UK = Arrays.copyOf(UK, size);
        UB = Arrays.copyOf(UB, size);
        UT = Arrays.copyOf(UT, size);
    }
}
