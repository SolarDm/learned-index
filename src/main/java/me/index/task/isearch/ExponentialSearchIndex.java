package me.index.task.isearch;

import me.index.libs.math.Sorted;
import me.index.task.Index;

public final class ExponentialSearchIndex implements Index {
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

        int l = 0, r = 1;

        while (l != keys.length - 1 && keys[r - 1] < key) {
            l = r - 1;
            r += r;

            if (r > keys.length) {
                r = keys.length;
            }
        }

        r--;

        if (l == keys.length - 1) {
            return r;
        }

        while (r - l > 1) {
            int m = (r + l) >> 1;

            if (keys[m] < key) {
                l = m;
            } else {
                r = m;
            }
        }

        return r;
    }

    @Override
    public int lower_bound(int key) {
        return Sorted.lowerBoundNear(keys, key, predict(key), maxError());
    }

    @Override
    public String id() {
        return "exponential";
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
