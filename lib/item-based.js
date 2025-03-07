/**
 * Item-Based Collaborative Filtering recommendation algorithm.
 * Uses cosine similarity between items (movies) instead of users.
 */
export class ItemBasedCF {
  /**
   * Constructor
   */
  constructor() {
    this.userRatings = null;
    this.movieRatings = null;
    this.itemSimilarities = {};
  }

  /**
   * Initialize the recommender with training data
   * @param {Object} userRatings - User ratings data
   * @param {Object} movieRatings - Movie ratings data
   */
  initialize(userRatings, movieRatings) {
    this.userRatings = userRatings;
    this.movieRatings = movieRatings;
  }

  /**
   * Compute cosine similarity between two movies.
   * @param {Object} movie1Ratings - Ratings of the first movie.
   * @param {Object} movie2Ratings - Ratings of the second movie.
   * @returns {number} - Cosine similarity score.
   */
  calculateItemSimilarity(movie1Ratings, movie2Ratings) {
    const commonUsers = Object.keys(movie1Ratings).filter(
      (user) => user in movie2Ratings,
    );

    if (commonUsers.length === 0) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (const userId of commonUsers) {
      dotProduct += movie1Ratings[userId] * movie2Ratings[userId];
      normA += movie1Ratings[userId] * movie1Ratings[userId];
      normB += movie2Ratings[userId] * movie2Ratings[userId];
    }

    if (normA === 0 || normB === 0) return 0;
    const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));

    const shrinkage = 10;
    return similarity * (commonUsers.length / (commonUsers.length + shrinkage));
  }

  /**
   * Calcuilate similarity between two movies (cached for performance)
   * @param {number} movie1Id - First movie ID
   * @param {number} movie2Id - Second movie ID
   * @returns {number} - Similarity score
   */
  calculateSimilarity(movie1Id, movie2Id) {
    const key = `${Math.min(movie1Id, movie2Id)}_${Math.max(movie1Id, movie2Id)}`;

    if (this.itemSimilarities[key] === undefined) {
      const movie1Ratings = this.movieRatings[movie1Id] || {};
      const movie2Ratings = this.movieRatings[movie2Id] || {};

      this.itemSimilarities[key] = this.calculateItemSimilarity(
        movie1Ratings,
        movie2Ratings,
      );
    }

    return this.itemSimilarities[key];
  }

  /**
   * Predict movie ratings for a user
   * @param {number} userId - User ID
   * @param {Object} knownRatings - Known ratings of the user
   * @param {Array<number>} moviesToPredict - List of movies to predict
   * @returns {Object} - Predicted ratings
   */
  predict(userId, knownRatings, moviesToPredict) {
    const predictions = {};
    const globalAverage = 3;

    for (const targetMovieId of moviesToPredict) {
      let weightedSum = 0;
      let similaritySum = 0;

      for (const ratedMovieId in knownRatings) {
        const similarity = this.calculateSimilarity(
          parseInt(ratedMovieId),
          targetMovieId,
        );

        if (similarity > 0) {
          const baseline = this.getAverageRating(parseInt(ratedMovieId));
          weightedSum += similarity * (knownRatings[ratedMovieId] - baseline);
          similaritySum += similarity;
        }
      }

      const targetBaseline = this.getAverageRating(targetMovieId);
      let prediction =
        similaritySum > 0
          ? targetBaseline + weightedSum / similaritySum
          : targetBaseline;

      prediction = Math.max(1, Math.min(5, prediction));

      if (prediction <= 0) {
        prediction = globalAverage;
      }

      predictions[targetMovieId] = prediction;
    }

    return predictions;
  }

  /**
   * Get the average rating for a specific movie
   * @param {number} movieId - Movie ID
   * @returns {number} - Average rating or 3 if not found
   */
  getAverageRating(movieId) {
    if (!this.movieRatings[movieId]) return 3;

    const ratings = Object.values(this.movieRatings[movieId]);
    if (ratings.length === 0) return 3;

    return ratings.reduce((sum, rating) => sum + rating, 0) / ratings.length;
  }
}
