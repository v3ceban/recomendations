/**
 * Matrix Factorization based recommender using Singular Value Decomposition (SVD)
 * This is an implementation of a simplified SVD approach for collaborative filtering.
 */
export class SVDRecommender {
  /**
   * Constructor
   * @param {Object} options - Configuration options
   * @param {number} options.features - Number of latent features (default: 50)
   * @param {number} options.iterations - Number of training iterations (default: 150)
   * @param {number} options.learningRate - Learning rate for gradient descent (default: 0.01)
   * @param {number} options.regularization - Regularization parameter (default: 0.02)
   * @param {number} options.biasRegularization - Regularization for bias terms (default: 0.005)
   * @param {boolean} options.useImplicitFeedback - Whether to use implicit feedback (default: true)
   */
  constructor(options = {}) {
    this.features = options.features || 50;
    this.iterations = options.iterations || 150;
    this.learningRate = options.learningRate || 0.01;
    this.regularization = options.regularization || 0.02;
    this.biasRegularization = options.biasRegularization || 0.005;
    this.useImplicitFeedback = options.useImplicitFeedback !== false;

    this.userRatings = null;
    this.movieRatings = null;
    this.userFeatures = {};
    this.movieFeatures = {};
    this.globalAverage = 0;
    this.userBiases = {};
    this.movieBiases = {};
    this.userRatingCounts = {};
    this.movieRatingCounts = {};
    this.confidenceFactors = {};
  }

  /**
   * Initialize the recommender with training data
   * @param {Object} userRatings - User ratings data
   * @param {Object} movieRatings - Movie ratings data
   */
  initialize(userRatings, movieRatings) {
    this.userRatings = userRatings;
    this.movieRatings = movieRatings;

    let totalRatings = 0;
    let ratingSum = 0;

    for (const userId in this.userRatings) {
      const userRatingObj = this.userRatings[userId];
      this.userRatingCounts[userId] = Object.keys(userRatingObj).length;

      for (const movieId in userRatingObj) {
        const rating = userRatingObj[movieId];
        ratingSum += rating;
        totalRatings++;

        this.movieRatingCounts[movieId] =
          (this.movieRatingCounts[movieId] || 0) + 1;
      }
    }

    this.globalAverage = ratingSum / Math.max(1, totalRatings);

    for (const movieId in this.movieRatingCounts) {
      const count = this.movieRatingCounts[movieId];
      this.confidenceFactors[movieId] = 1 + 1 / (1 + Math.exp(-count / 50 + 3));
    }

    for (const userId in this.userRatings) {
      const avgScale = Math.sqrt(this.globalAverage / this.features);
      this.userFeatures[userId] = Array(this.features)
        .fill()
        .map(() => avgScale * (Math.random() * 0.1 + 0.95));

      const userAvg =
        Object.values(this.userRatings[userId]).reduce((a, b) => a + b, 0) /
        this.userRatingCounts[userId];
      this.userBiases[userId] = 0.1 * (userAvg - this.globalAverage);
    }

    for (const movieId in this.movieRatings) {
      const avgScale = Math.sqrt(this.globalAverage / this.features);
      this.movieFeatures[movieId] = Array(this.features)
        .fill()
        .map(() => avgScale * (Math.random() * 0.1 + 0.95));

      const ratings = Object.values(this.movieRatings[movieId]);
      const movieAvg = ratings.reduce((a, b) => a + b, 0) / ratings.length;
      this.movieBiases[movieId] = 0.1 * (movieAvg - this.globalAverage);
    }

    this.train();
  }

  /**
   * Train the SVD model using stochastic gradient descent
   * with improved regularization and adaptive learning rates
   */
  train() {
    let currentLearningRate = this.learningRate;
    const learningRateDecay = 0.9;
    let previousRMSE = Number.MAX_VALUE;

    for (let iter = 0; iter < this.iterations; iter++) {
      let totalError = 0;
      let n = 0;

      for (const userId in this.userRatings) {
        const userRatingObj = this.userRatings[userId];
        const userFeatures = this.userFeatures[userId];

        for (const movieId in userRatingObj) {
          const actualRating = userRatingObj[movieId];
          const movieFeatures = this.movieFeatures[movieId];

          const confidence = this.useImplicitFeedback
            ? this.confidenceFactors[movieId] || 1
            : 1;

          let predictedRating =
            this.globalAverage +
            this.userBiases[userId] +
            this.movieBiases[movieId];

          for (let f = 0; f < this.features; f++) {
            predictedRating += userFeatures[f] * movieFeatures[f];
          }

          predictedRating = Math.max(1, Math.min(5, predictedRating));

          const error = (actualRating - predictedRating) * confidence;
          totalError += error * error;
          n++;

          this.userBiases[userId] +=
            currentLearningRate *
            (error - this.biasRegularization * this.userBiases[userId]);
          this.movieBiases[movieId] +=
            currentLearningRate *
            (error - this.biasRegularization * this.movieBiases[movieId]);

          const userRegTerm =
            this.regularization / Math.sqrt(this.userRatingCounts[userId] || 1);
          const movieRegTerm =
            this.regularization /
            Math.sqrt(this.movieRatingCounts[movieId] || 1);

          for (let f = 0; f < this.features; f++) {
            const userFeatureOld = userFeatures[f];
            const movieFeatureOld = movieFeatures[f];

            userFeatures[f] +=
              currentLearningRate *
              (error * movieFeatureOld - userRegTerm * userFeatureOld);
            movieFeatures[f] +=
              currentLearningRate *
              (error * userFeatureOld - movieRegTerm * movieFeatureOld);
          }
        }
      }

      const rmse = Math.sqrt(totalError / n);

      if (rmse >= previousRMSE * 0.997) {
        currentLearningRate *= learningRateDecay;
      }
      previousRMSE = rmse;

      if (rmse < 0.001 || currentLearningRate < 1e-6) {
        break;
      }
    }
  }

  /**
   * Calculate average rating for a specific user or movie
   * @param {Object} ratings - Ratings object
   * @returns {number} - Average rating
   */
  getAverageRating(ratings) {
    const values = Object.values(ratings);
    return values.length > 0
      ? values.reduce((sum, r) => sum + r, 0) / values.length
      : this.globalAverage;
  }

  /**
   * Calculate the dot product of two feature vectors
   * @param {Array} userFeatures - User feature vector
   * @param {Array} movieFeatures - Movie feature vector
   * @returns {number} - Dot product
   */
  dotProduct(userFeatures, movieFeatures) {
    let sum = 0;
    for (let i = 0; i < this.features; i++) {
      sum += userFeatures[i] * movieFeatures[i];
    }
    return sum;
  }

  /**
   * Initialize features for a new user based on their known ratings
   * using folding-in method with adaptive fitting
   * @param {number} userId - User ID
   * @param {Object} knownRatings - Known ratings of the user
   * @returns {Object} - Features and bias for the new user
   */
  initializeNewUserFeatures(userId, knownRatings) {
    const ratings = Object.values(knownRatings);
    const numRatings = ratings.length;

    const userAvgRating =
      numRatings > 0
        ? ratings.reduce((sum, r) => sum + r, 0) / numRatings
        : this.globalAverage;

    const userBias =
      numRatings > 2
        ? 0.15 * (userAvgRating - this.globalAverage)
        : 0.05 * (userAvgRating - this.globalAverage);

    const avgScale = Math.sqrt(this.globalAverage / this.features);
    const userFeatures = Array(this.features)
      .fill()
      .map((_, i) => {
        const factorScale = avgScale * (1 - 0.3 * (i / this.features));
        return factorScale * (Math.random() * 0.2 + 0.9);
      });

    const iterations = Math.min(30, Math.max(15, numRatings * 2));

    let learningRate = 0.025;

    let prevError = Number.MAX_VALUE;

    for (let iter = 0; iter < iterations; iter++) {
      let totalError = 0;
      let count = 0;

      const movieIds = Object.keys(knownRatings);
      for (let i = movieIds.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [movieIds[i], movieIds[j]] = [movieIds[j], movieIds[i]];
      }

      for (const movieId of movieIds) {
        if (this.movieFeatures[movieId]) {
          const actualRating = knownRatings[movieId];
          const movieFeatures = this.movieFeatures[movieId];
          const movieBias = this.movieBiases[movieId] || 0;

          let predictedRating = this.globalAverage + userBias + movieBias;
          for (let f = 0; f < this.features; f++) {
            predictedRating += userFeatures[f] * movieFeatures[f];
          }

          predictedRating = Math.max(1, Math.min(5, predictedRating));

          const error = actualRating - predictedRating;
          totalError += error * error;
          count++;

          const regTerm = this.regularization / Math.pow(numRatings, 0.4);

          for (let f = 0; f < this.features; f++) {
            const featureWeight = 1.0 - 0.2 * (f / this.features);
            userFeatures[f] +=
              learningRate *
              featureWeight *
              (error * movieFeatures[f] - regTerm * userFeatures[f]);
          }
        }
      }

      const iterError = count > 0 ? Math.sqrt(totalError / count) : 0;

      if (iterError >= prevError * 0.98) {
        learningRate *= 0.75;
      }
      prevError = iterError;

      if (iterError < 0.001 || learningRate < 0.001) {
        break;
      }
    }

    return { features: userFeatures, bias: userBias };
  }

  /**
   * Predict movie ratings for a user with improved interpolation
   * @param {number} userId - User ID
   * @param {Object} knownRatings - Known ratings of the user
   * @param {Array<number>} moviesToPredict - List of movies to predict
   * @returns {Object} - Predicted ratings
   */
  predict(userId, knownRatings, moviesToPredict) {
    const predictions = {};

    const { features: userFeatures, bias: userBias } =
      this.initializeNewUserFeatures(userId, knownRatings);

    const userAvgRating = this.getAverageRating(knownRatings);

    const confidenceFactor = Math.min(1, Object.keys(knownRatings).length / 15);

    for (const movieId of moviesToPredict) {
      if (this.movieFeatures[movieId]) {
        let svdPrediction =
          this.globalAverage + userBias + (this.movieBiases[movieId] || 0);
        for (let f = 0; f < this.features; f++) {
          svdPrediction += userFeatures[f] * this.movieFeatures[movieId][f];
        }

        const movieAvgRating = this.movieRatings[movieId]
          ? this.getAverageRating(this.movieRatings[movieId])
          : this.globalAverage;

        const userBiasFactor = Math.min(0.3, 0.1 + 0.2 * confidenceFactor);
        const baselinePrediction =
          movieAvgRating +
          userBiasFactor * (userAvgRating - this.globalAverage);

        const movieConfidence = this.confidenceFactors[movieId] || 1;
        const blendFactor = 0.7 * confidenceFactor * movieConfidence;

        const finalPrediction =
          blendFactor * svdPrediction + (1 - blendFactor) * baselinePrediction;

        predictions[movieId] = Math.max(1, Math.min(5, finalPrediction));
      } else {
        const userBias = userAvgRating - this.globalAverage;
        predictions[movieId] = Math.max(
          1,
          Math.min(5, this.globalAverage + 0.2 * userBias),
        );
      }
    }

    return predictions;
  }
}
