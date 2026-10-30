package me.index.task.inet.model;

import me.index.libs.math.Maths;
import me.index.task.inet.Init;
import me.index.task.inet.loss.Loss;

import java.util.Arrays;
import java.util.Random;


public abstract class AbstractNet implements Model {
    private final double learningRate;
    private final int batchSize;
    private final Loss loss;
    private final Random rnd;
    private final String activation;
    private final int layers;
    private final double[][][] weights;
    private final double[][] biases;
    private final double[][] activations;
    private final double[][] preActivations;
    private final double[][] deltas;
    private final double[][][] weightGradients;
    private final double[][] biasGradients;

    protected AbstractNet(int[] layerSizes, double learningRate, int batchSize, Loss loss, Random rnd,
                          Init init, String activation) {
        if (layerSizes.length < 2 || layerSizes[0] != 1 || layerSizes[layerSizes.length - 1] != 1) {
            throw new IllegalArgumentException("input and output layers must have size 1, got "
                    + Arrays.toString(layerSizes));
        }
        if (!(learningRate > 0) || !Double.isFinite(learningRate)) {
            throw new IllegalArgumentException("learningRate must be finite and positive, got " + learningRate);
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive, got " + batchSize);
        }
        if (loss == null || rnd == null) {
            throw new IllegalArgumentException("loss and rnd must not be null");
        }
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.loss = loss;
        this.rnd = rnd;
        this.activation = activation;
        this.layers = layerSizes.length - 1;
        this.weights = new double[layers][][];
        this.biases = new double[layers][];
        this.activations = new double[layers + 1][];
        this.preActivations = new double[layers][];
        this.deltas = new double[layers][];
        this.weightGradients = new double[layers][][];
        this.biasGradients = new double[layers][];
        activations[0] = new double[layerSizes[0]];
        for (int l = 0; l < layers; l++) {
            int in = layerSizes[l];
            int out = layerSizes[l + 1];
            weights[l] = Maths.gaussian(out, in, init.scale(in, out), rnd);
            biases[l] = new double[out];
            activations[l + 1] = new double[out];
            preActivations[l] = new double[out];
            deltas[l] = new double[out];
            weightGradients[l] = new double[out][in];
            biasGradients[l] = new double[out];
        }
    }

    protected abstract double activate(double z);

    protected abstract double derivative(double z, double value);

    @Override
    public double predict(double x) {
        forward(x);
        return activations[layers][0];
    }

    @Override
    public double train(double[] xs, double[] ys, int epochs) {
        if (xs.length != ys.length) {
            throw new IllegalArgumentException("xs and ys differ in length: " + xs.length + " vs " + ys.length);
        }
        if (epochs < 0) {
            throw new IllegalArgumentException("epochs must not be negative, got " + epochs);
        }
        if (xs.length == 0 || epochs == 0) {
            return Double.NaN;
        }
        int[] order = new int[xs.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        for (int epoch = 0; epoch < epochs; epoch++) {
            shuffle(order);
            int count = 0;
            for (int i : order) {
                step(xs[i], ys[i]);
                count++;
                if (count == batchSize) {
                    update(count);
                    count = 0;
                }
            }
            if (count > 0) {
                update(count);
            }
        }
        return meanSquaredError(xs, ys);
    }

    private void step(double x, double y) {
        forward(x);
        backward(y);
        accumulateGradients();
    }

    private void forward(double x) {
        activations[0][0] = x;
        for (int l = 0; l < layers; l++) {
            double[] input = activations[l];
            double[] output = activations[l + 1];
            for (int i = 0; i < output.length; i++) {
                double z = 0.0;
                for (int j = 0; j < input.length; j++) {
                    z += weights[l][i][j] * input[j];
                }
                z += biases[l][i];
                preActivations[l][i] = z;
                output[i] = l == layers - 1 ? z : activate(z);
            }
        }
    }

    private void backward(double y) {
        deltas[layers - 1][0] = loss.gradient(activations[layers][0], y);
        for (int l = layers - 2; l >= 0; l--) {
            for (int i = 0; i < deltas[l].length; i++) {
                double back = 0.0;
                for (int j = 0; j < deltas[l + 1].length; j++) {
                    back += weights[l + 1][j][i] * deltas[l + 1][j];
                }
                deltas[l][i] = back * derivative(preActivations[l][i], activations[l + 1][i]);
            }
        }
    }

    private void accumulateGradients() {
        for (int l = 0; l < layers; l++) {
            double[] input = activations[l];
            for (int i = 0; i < deltas[l].length; i++) {
                double delta = deltas[l][i];
                biasGradients[l][i] += delta;
                for (int j = 0; j < input.length; j++) {
                    weightGradients[l][i][j] += delta * input[j];
                }
            }
        }
    }

    private void update(int count) {
        // нормализуем на батч
        double scale = learningRate / count;
        for (int l = 0; l < layers; l++) {
            for (int i = 0; i < weights[l].length; i++) {
                biases[l][i] -= scale * biasGradients[l][i];
                biasGradients[l][i] = 0.0;
                for (int j = 0; j < weights[l][i].length; j++) {
                    weights[l][i][j] -= scale * weightGradients[l][i][j];
                    weightGradients[l][i][j] = 0.0;
                }
            }
        }
    }

    private void shuffle(int[] order) {
        for (int i = order.length - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int swapped = order[i];
            order[i] = order[j];
            order[j] = swapped;
        }
    }

    private double meanSquaredError(double[] xs, double[] ys) {
        double sum = 0.0;
        for (int i = 0; i < xs.length; i++) {
            double error = predict(xs[i]) - ys[i];
            sum += error * error;
        }
        return sum / xs.length;
    }

    @Override
    public String id() {
        return activation + "/" + loss.id();
    }

    @Override
    public long sizeInBytes() {
        long doubles = activations[0].length;
        for (int l = 0; l < layers; l++) {
            doubles += (long) weights[l].length * weights[l][0].length
                    + biases[l].length + activations[l + 1].length;
        }
        return doubles * Double.BYTES;
    }
}
