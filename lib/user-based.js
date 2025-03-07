/**
 * User-Based Collaborative Filtering recommendation algorithm.
 * Implements both Cosine similarity and Pearson correlation methods.
 */
export class UserBasedCF {
  /**
   * Constructor
   * @param {string} similarityMethod - The similarity method to use ('cosine' or 'pearson')
   */
  constructor(similarityMethod = "cosine") {
    this.similarityMethod = similarityMethod;
    this.userRatings = null;
    this.movieRatings = null;
    this.userAverageRatings = {};
  }

  /**
   * Initialize the recommender with training data
   * @param {Object} userRatings - User ratings data
   * @param {Object} movieRatings - Movie ratings data
   */
  initialize(userRatings, movieRatings) {
    this.userRatings = userRatings;
    this.movieRatings = movieRatings;

    for (const userId in this.userRatings) {
      const ratings = Object.values(this.userRatings[userId]);
      if (ratings.length > 0) {
        this.userAverageRatings[userId] =
          ratings.reduce((sum, rating) => sum + rating, 0) / ratings.length;
      }
    }
  }

  /**
   * Compute cosine similarity between two users.
   * @param {Object} user1Ratings - Ratings of the first user.
   * @param {Object} user2Ratings - Ratings of the second user.
   * @returns {number} - Cosine similarity score.
   */
  cosineSimilarity(user1Ratings, user2Ratings) {
    let dotProduct = 0,
      normA = 0,
      normB = 0;

    const commonMovies = Object.keys(user1Ratings).filter(
      (movie) => movie in user2Ratings,
    );

    if (commonMovies.length === 0) return 0;

    for (const movie of commonMovies) {
      dotProduct += user1Ratings[movie] * user2Ratings[movie];
      normA += Math.pow(user1Ratings[movie], 2);
      normB += Math.pow(user2Ratings[movie], 2);
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Compute Pearson correlation between two users.
   * @param {Object} user1Ratings - Ratings of the first user.
   * @param {Object} user2Ratings - Ratings of the second user.
   * @param {number} user1Id - ID of the first user
   * @param {number} user2Id - ID of the second user
   * @returns {number} - Pearson correlation coefficient.
   */
  pearsonCorrelation(user1Ratings, user2Ratings, user1Id, user2Id) {
    const commonMovies = Object.keys(user1Ratings).filter(
      (movie) => movie in user2Ratings,
    );

    if (commonMovies.length === 0) return 0;

    const user1Avg =
      this.userAverageRatings[user1Id] ||
      Object.values(user1Ratings).reduce((sum, r) => sum + r, 0) /
        Object.values(user1Ratings).length;

    const user2Avg =
      this.userAverageRatings[user2Id] ||
      Object.values(user2Ratings).reduce((sum, r) => sum + r, 0) /
        Object.values(user2Ratings).length;

    let numerator = 0;
    let denominator1 = 0;
    let denominator2 = 0;

    for (const movie of commonMovies) {
      const user1Deviation = user1Ratings[movie] - user1Avg;
      const user2Deviation = user2Ratings[movie] - user2Avg;

      numerator += user1Deviation * user2Deviation;
      denominator1 += Math.pow(user1Deviation, 2);
      denominator2 += Math.pow(user2Deviation, 2);
    }

    if (denominator1 === 0 || denominator2 === 0) return 0;
    return numerator / (Math.sqrt(denominator1) * Math.sqrt(denominator2));
  }

  /**
   * Calculate similarity between two users using the selected method
   * @param {Object} user1Ratings - Ratings of the first user
   * @param {Object} user2Ratings - Ratings of the second user
   * @param {number} user1Id - ID of the first user
   * @param {number} user2Id - ID of the second user
   * @returns {number} - Similarity score
   */
  calculateSimilarity(user1Ratings, user2Ratings, user1Id, user2Id) {
    if (this.similarityMethod === "pearson") {
      return this.pearsonCorrelation(
        user1Ratings,
        user2Ratings,
        user1Id,
        user2Id,
      );
    } else {
      return this.cosineSimilarity(user1Ratings, user2Ratings);
    }
  }

  /**
   * Predict movie ratings for a user
   * @param {number} userId - User ID
   * @param {Object} knownRatings - Known ratings of the user
   * @param {Array<number>} moviesToPredict - List of movies to predict
   * @returns {Object} - Predicted ratings
   */
  predict(userId, knownRatings, moviesToPredict) {
    const similarities = {};
    const userAvg =
      Object.values(knownRatings).reduce((sum, r) => sum + r, 0) /
      Object.values(knownRatings).length;

    if (!this.userAverageRatings[userId]) {
      this.userAverageRatings[userId] = userAvg;
    }

    for (const trainUserId in this.userRatings) {
      similarities[trainUserId] = this.calculateSimilarity(
        knownRatings,
        this.userRatings[trainUserId],
        userId,
        Number(trainUserId),
      );
    }

    const predictions = {};
    for (const movieId of moviesToPredict) {
      let weightedSum = 0;
      let similaritySum = 0;

      for (const trainUserId in this.userRatings) {
        if (this.userRatings[trainUserId][movieId]) {
          const similarity = similarities[trainUserId];
          if (similarity > 0) {
            const trainUserAvg = this.userAverageRatings[trainUserId];
            const trainUserRating = this.userRatings[trainUserId][movieId];

            if (this.similarityMethod === "pearson") {
              weightedSum += similarity * (trainUserRating - trainUserAvg);
            } else {
              weightedSum += similarity * trainUserRating;
            }
            similaritySum += Math.abs(similarity);
          }
        }
      }

      if (similaritySum > 0) {
        let predictedRating;
        if (this.similarityMethod === "pearson") {
          predictedRating = userAvg + weightedSum / similaritySum;
        } else {
          predictedRating = weightedSum / similaritySum;
        }
        predictions[movieId] = Math.max(1, Math.min(5, predictedRating));
      } else {
        const avgRating = this.getAverageRating(movieId);
        predictions[movieId] = avgRating ? Math.max(1, avgRating) : 3;
      }
    }

    return predictions;
  }

  /**
   * Get the average rating for a specific movie
   * @param {number} movieId - Movie ID
   * @returns {number} - Average rating or null if not found
   */
  getAverageRating(movieId) {
    if (!this.movieRatings[movieId]) return null;

    const ratings = Object.values(this.movieRatings[movieId]);
    if (ratings.length === 0) return null;

    const average =
      ratings.reduce((sum, rating) => sum + rating, 0) / ratings.length;
    return average > 0 ? average : null;
  }
}
