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
        int lo = 0;
        int hi = 1;
        while (hi < keys.length - 1 && keys[hi] < key) {
            lo = hi + 1;
            hi = (int) Math.min(2L * hi, keys.length - 1);
        }
        return Math.min(Sorted.lowerBound(keys, lo, hi + 1, key), keys.length - 1);
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
