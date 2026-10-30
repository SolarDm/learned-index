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
        if (keys.length == 0 || keys[0] >= key) {
            return 0;
        }

        int l = 0, r = keys.length - 1;

        if (keys[r] < key) {
            return r + 1;
        }

        while (keys[r] >= key && keys[l] <= key && keys[r] != keys[l]) {
            int m = l + (int) (((long) key - keys[l]) * (long) (r - l) / ((long) keys[r] - keys[l]));

            if (keys[m] < key) {
                l = m + 1;
            } else if (keys[m] > key) {
                r = m - 1;
            } else {
                return m;
            }
        }

        if (keys[r] >= key) {
            return r;
        } else {
            return r + 1;
        }
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
