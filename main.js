#!/usr/bin/env node

import fs from "fs";
import { readFile } from "./lib/read-file.js";
import { UserBasedCF } from "./lib/user-based.js";
import { ExtendedUserBasedCF } from "./lib/user-based-extensions.js";
import { ItemBasedCF } from "./lib/item-based.js";
import { HybridCF } from "./lib/hybrid.js";
import { SVDRecommender } from "./lib/svd.js";
import { HybridSVDCF } from "./lib/hybrid-svd.js";

/**
 * Parse training data lines into a structured format.
 * @param {Array<string>} trainLines - Array of lines from the training data file.
 * @returns {Promise<Object>} - Parsed user rating data and movie rating data.
 */
async function loadTrainingData(trainLines) {
  const userRatings = {};
  const movieRatings = {};

  for (const line of trainLines) {
    const [user, movie, rating] = line.split(" ").map(Number);

    if (!userRatings[user]) userRatings[user] = {};
    userRatings[user][movie] = rating;

    if (!movieRatings[movie]) movieRatings[movie] = {};
    movieRatings[movie][user] = rating;
  }

  return { userRatings, movieRatings };
}

/**
 * Group test data lines by user.
 * In the test file, ratings with value 0 indicate missing values that need prediction.
 * The output is an array of objects for each test user with known ratings and movies to predict.
 * @param {Array<string>} testDataLines - Array of lines from the test data file.
 * @returns {Promise<Array<Object>>} - Array of test user objects.
 * Each object contains:
 * - user: user id
 * - knownRatings: object mapping movie id to rating
 * - missing: array of movie ids that require prediction */
async function loadTestData(testDataLines) {
  const testUsers = [];
  let currentUser = null;
  let currentUserId = null;

  for (const line of testDataLines) {
    if (!line.trim()) continue;
    const [user, movie, rating] = line.split(" ").map(Number);
    if (currentUserId !== user) {
      if (currentUser) testUsers.push(currentUser);
      currentUserId = user;
      currentUser = { user, knownRatings: {}, missing: [] };
    }
    if (rating === 0) {
      currentUser.missing.push(movie);
    } else {
      currentUser.knownRatings[movie] = rating;
    }
  }
  if (currentUser) testUsers.push(currentUser);
  return testUsers;
}

/**
 * Print test data with predictions replacing zeros
 * @param {Array<string>} testDataLines - Array of lines from the test data file.
 * @param {Object} predictions - Object containing predicted ratings
 * @param {number} userId - ID of the user whose predictions to print
 */
async function printPredictedData(testDataLines, predictions, userId) {
  for (const line of testDataLines) {
    if (!line.trim()) continue;
    const [user, movie, rating] = line.split(" ").map(Number);
    if (user === userId && rating === 0) {
      const predictedRating = Math.round(predictions[movie]);
      console.log(`${user} ${movie} ${predictedRating}`);
    }
  }
}

/**
 * Main function to run the recommendation system
 */
(async () => {
  const args = process.argv.slice(2);
  if (args.length !== 2) {
    console.error("Usage: main.js <test_file> <algorithm>\n");
    console.error("Available algorithms:");
    console.error("user-cosine - User-based CF with Cosine similarity");
    console.error("user-pearson - User-based CF with Pearson correlation");
    console.error("user-iuf - User-based CF with inverse user frequency");
    console.error("user-case - User-based CF with case modification");
    console.error("item - Item-based CF with Cosine similarity");
    console.error("hybrid - Custom hybrid recommendation algorithm");
    console.error("svd - Matrix factorization with SVD");
    console.error("hybrid-svd - Hybrid recommendation algorithm with SVD");
    process.exit(1);
  }

  try {
    const testDataPath = args[0];

    if (!testDataPath.endsWith(".txt") || !fs.existsSync(testDataPath)) {
      console.error("Test file doesn't exist or is in invalid file format");
      process.exit(1);
    }

    const algorithmType = args[1];

    if (
      ![
        "user-cosine",
        "user-pearson",
        "user-iuf",
        "user-case",
        "item",
        "hybrid",
        "svd",
        "hybrid-svd",
      ].includes(algorithmType)
    ) {
      console.error("Invalid algorithm type");
      process.exit(1);
    }

    const trainDataLines = await readFile("./train.txt");
    const testDataLines = await readFile(testDataPath);
    const { userRatings, movieRatings } =
      await loadTrainingData(trainDataLines);
    const testUsers = await loadTestData(testDataLines);

    let recommender;
    switch (algorithmType) {
      case "user-cosine":
        recommender = new UserBasedCF("cosine");
        break;
      case "user-pearson":
        recommender = new UserBasedCF("pearson");
        break;
      case "user-iuf":
        recommender = new ExtendedUserBasedCF("iuf");
        break;
      case "user-case":
        recommender = new ExtendedUserBasedCF("case");
        break;
      case "item":
        recommender = new ItemBasedCF();
        break;
      case "hybrid":
        recommender = new HybridCF();
        break;
      case "svd":
        recommender = new SVDRecommender();
        break;
      case "hybrid-svd":
        recommender = new HybridSVDCF();
        break;
      default:
        console.error(`Unknown algorithm: ${algorithmType}`);
        process.exit(1);
    }

    recommender.initialize(userRatings, movieRatings);

    for (const testUser of testUsers) {
      const predictions = recommender.predict(
        testUser.user,
        testUser.knownRatings,
        testUser.missing,
      );
      await printPredictedData(testDataLines, predictions, testUser.user);
    }
  } catch (error) {
    console.error("Error:", error.message);
  }
})();
