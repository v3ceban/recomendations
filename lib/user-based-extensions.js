import { UserBasedCF } from "./user-based.js";

/**
 * Extended User-Based Collaborative Filtering with advanced features:
 * 1. Inverse User Frequency - Weights rare item ratings more strongly
 * 2. Case Modification - Adjusts similarity based on number of common items
 */
export class ExtendedUserBasedCF extends UserBasedCF {
  /**
   * Constructor
   * @param {string} extensionType - The extension type ('iuf' or 'case')
   */
  constructor(extensionType = "iuf") {
    super("pearson");
    this.extensionType = extensionType;
    this.iufValues = {};
    this.movieFrequency = {};
    this.totalUsers = 0;
  }

  /**
   * Initialize the recommender with training data
   * @param {Object} userRatings - User ratings data
   * @param {Object} movieRatings - Movie ratings data
   */
  initialize(userRatings, movieRatings) {
    super.initialize(userRatings, movieRatings);

    this.totalUsers = Object.keys(userRatings).length;

    if (this.extensionType === "iuf") {
      for (const userId in userRatings) {
        for (const movieId in userRatings[userId]) {
          this.movieFrequency[movieId] =
            (this.movieFrequency[movieId] || 0) + 1;
        }
      }

      for (const movieId in this.movieFrequency) {
        this.iufValues[movieId] = Math.log(
          this.totalUsers / this.movieFrequency[movieId],
        );
      }
    }
  }

  /**
   * Calculate similarity with IUF weighting
   * @param {Object} user1Ratings - Ratings of the first user
   * @param {Object} user2Ratings - Ratings of the second user
   * @returns {number} - IUF-weighted similarity
   */
  calculateIUFSimilarity(user1Ratings, user2Ratings) {
    const commonMovies = Object.keys(user1Ratings).filter(
      (movie) => movie in user2Ratings,
    );

    if (commonMovies.length === 0) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (const movie of commonMovies) {
      const iufWeight = this.iufValues[movie] || 1;
      dotProduct += user1Ratings[movie] * user2Ratings[movie] * iufWeight;
      normA += Math.pow(user1Ratings[movie] * iufWeight, 2);
      normB += Math.pow(user2Ratings[movie] * iufWeight, 2);
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Calculate similarity with case modification
   * @param {Object} user1Ratings - Ratings of the first user
   * @param {Object} user2Ratings - Ratings of the second user
   * @param {number} user1Id - ID of the first user
   * @param {number} user2Id - ID of the second user
   * @returns {number} - Case-modified similarity
   */
  calculateCaseModifiedSimilarity(
    user1Ratings,
    user2Ratings,
    user1Id,
    user2Id,
  ) {
    const correlation = this.pearsonCorrelation(
      user1Ratings,
      user2Ratings,
      user1Id,
      user2Id,
    );

    const commonMovies = Object.keys(user1Ratings).filter(
      (movie) => movie in user2Ratings,
    );

    const n = commonMovies.length;
    const threshold = 50;

    return correlation * Math.pow(Math.min(n, threshold) / threshold, 1.25);
  }

  /**
   * Calculate similarity between two users using the selected extension method
   * @param {Object} user1Ratings - Ratings of the first user
   * @param {Object} user2Ratings - Ratings of the second user
   * @param {number} user1Id - ID of the first user
   * @param {number} user2Id - ID of the second user
   * @returns {number} - Similarity score
   */
  calculateSimilarity(user1Ratings, user2Ratings, user1Id, user2Id) {
    if (this.extensionType === "iuf") {
      return this.calculateIUFSimilarity(user1Ratings, user2Ratings);
    } else if (this.extensionType === "case") {
      return this.calculateCaseModifiedSimilarity(
        user1Ratings,
        user2Ratings,
        user1Id,
        user2Id,
      );
    }
    return super.calculateSimilarity(
      user1Ratings,
      user2Ratings,
      user1Id,
      user2Id,
    );
  }
}
