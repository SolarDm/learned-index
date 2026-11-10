package me.index.task.isearch;

import me.index.libs.math.Sorted;
import me.index.task.Index;

public final class InterpolationSearchIndex implements Index {
    private int[] keys;

    @Override
    public void build(int[] keys) {
        Sorted.checkKeys(keys);
        this.keys = keys;
    }

    @Override
    public int predict(int key) {
        int lo = 0;
        int hi = keys.length - 1;
        while (lo < hi && keys[lo] < key && key <= keys[hi]) {
            final int mid = lo + (int) (((long) key - keys[lo]) * (hi - lo)
                    / ((long) keys[hi] - keys[lo]));
            if (keys[mid] == key) {
                return mid;
            }
            if (keys[mid] < key) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (keys[hi] < key) {
            return Math.min(hi + 1, keys.length - 1);
        }
        return lo;
    }

    @Override
    public int lower_bound(int key) {
        return Sorted.lowerBoundNear(keys, key, predict(key), maxError());
    }

    @Override
    public String id() {
        return "interpolation";
    }

    @Override
    public int maxError() {
        return 0;
    }

    @Override
    public long sizeInBytes() {
        return 0;
    }
}
