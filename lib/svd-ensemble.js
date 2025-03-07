import { SVDRecommender } from "./svd.js";

/**
 * Ensemble SVD recommender that combines multiple SVD models
 * to achieve better prediction accuracy
 */
export class EnsembleSVDRecommender {
  /**
   * Constructor for ensemble recommender
   */
  constructor() {
    this.models = [
      new SVDRecommender({
        features: 40,
        iterations: 120,
        learningRate: 0.008,
        regularization: 0.018,
        biasRegularization: 0.004,
      }),
      new SVDRecommender({
        features: 60,
        iterations: 150,
        learningRate: 0.01,
        regularization: 0.015,
        biasRegularization: 0.003,
      }),
      new SVDRecommender({
        features: 35,
        iterations: 130,
        learningRate: 0.006,
        regularization: 0.02,
        biasRegularization: 0.006,
        useImplicitFeedback: true,
      }),
      new SVDRecommender({
        features: 50,
        iterations: 140,
        learningRate: 0.012,
        regularization: 0.014,
        biasRegularization: 0.0035,
        useImplicitFeedback: true,
      }),
    ];

    this.modelWeights = [0.25, 0.3, 0.2, 0.25];

    this.userRatings = null;
    this.movieRatings = null;
    this.globalAverage = 0;
  }

  /**
   * Initialize all models with training data
   */
  initialize(userRatings, movieRatings) {
    this.userRatings = userRatings;
    this.movieRatings = movieRatings;

    let totalRatings = 0;
    let ratingSum = 0;

    for (const userId in userRatings) {
      for (const movieId in userRatings[userId]) {
        ratingSum += userRatings[userId][movieId];
        totalRatings++;
      }
    }

    this.globalAverage = totalRatings > 0 ? ratingSum / totalRatings : 3;

    for (const model of this.models) {
      model.initialize(userRatings, movieRatings);
    }
  }

  /**
   * Predict ratings using weighted ensemble
   */
  predict(userId, knownRatings, moviesToPredict) {
    const predictions = {};

    const modelPredictions = this.models.map((model) =>
      model.predict(userId, knownRatings, moviesToPredict),
    );

    const numRatings = Object.keys(knownRatings).length;

    let adjustedWeights = [...this.modelWeights];

    if (numRatings < 5) {
      adjustedWeights[0] += 0.1;
      adjustedWeights[1] -= 0.05;
      adjustedWeights[2] += 0.05;
      adjustedWeights[3] -= 0.1;
    }

    const weightSum = adjustedWeights.reduce((sum, weight) => sum + weight, 0);
    adjustedWeights = adjustedWeights.map((w) => w / weightSum);

    for (const movieId of moviesToPredict) {
      let weightedSum = 0;
      let validPredictions = 0;

      for (let i = 0; i < this.models.length; i++) {
        if (modelPredictions[i][movieId] !== undefined) {
          weightedSum += adjustedWeights[i] * modelPredictions[i][movieId];
          validPredictions++;
        }
      }

      if (validPredictions > 0) {
        predictions[movieId] = Math.max(1, Math.min(5, weightedSum));
      } else {
        const userAvg =
          Object.values(knownRatings).reduce((a, b) => a + b, 0) /
          Math.max(1, Object.values(knownRatings).length);
        predictions[movieId] = Math.max(
          1,
          Math.min(
            5,
            this.globalAverage + 0.1 * (userAvg - this.globalAverage),
          ),
        );
      }
    }

    return predictions;
  }
}
