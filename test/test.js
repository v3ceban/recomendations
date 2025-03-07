#!/usr/bin/env node

import fs from "fs";
import path from "path";
import { readFile } from "../lib/read-file.js";
import { UserBasedCF } from "../lib/user-based.js";
import { ExtendedUserBasedCF } from "../lib/user-based-extensions.js";
import { ItemBasedCF } from "../lib/item-based.js";
import { HybridCF } from "../lib/hybrid.js";
import { HybridSVDCF } from "../lib/hybrid-svd.js";
import { SVDRecommender } from "../lib/svd.js";

/**
 * Parse training data lines into a structured format.
 * @param {Array<string>} trainLines - Array of lines from the training data file.
 * @returns {Object} - Parsed user rating data and movie rating data.
 */
function loadTrainingData(trainLines) {
  const userRatings = {};
  const movieRatings = {};

  for (const line of trainLines) {
    if (!line.trim()) continue;
    const [user, movie, rating] = line.split(" ").map(Number);

    if (!userRatings[user]) userRatings[user] = {};
    userRatings[user][movie] = rating;

    if (!movieRatings[movie]) movieRatings[movie] = {};
    movieRatings[movie][user] = rating;
  }

  return { userRatings, movieRatings };
}

/**
 * Split a test file into validation and test sets
 * @param {string} testFilePath - Path to the test file
 * @param {string} [baseFileName] - Optional base name for output files
 * @returns {Promise<Object>} - Paths to created files
 */
async function splitTestFile(testFilePath, baseFileName) {
  const testLines = await readFile(testFilePath);

  const testUsers = {};

  for (const line of testLines) {
    if (!line.trim()) continue;
    const [user, movie, rating] = line.split(" ").map(Number);

    if (!testUsers[user]) {
      testUsers[user] = [];
    }

    testUsers[user].push({ user, movie, rating });
  }

  const userIds = Object.keys(testUsers);
  const validationUsers = [];
  const testUsers2 = [];

  for (let i = 0; i < userIds.length; i++) {
    if (i % 2 === 0) {
      validationUsers.push(userIds[i]);
    } else {
      testUsers2.push(userIds[i]);
    }
  }

  const validationLines = [];
  const testLines2 = [];

  for (const userId of validationUsers) {
    for (const entry of testUsers[userId]) {
      validationLines.push(`${entry.user} ${entry.movie} ${entry.rating}`);
    }
  }

  for (const userId of testUsers2) {
    for (const entry of testUsers[userId]) {
      testLines2.push(`${entry.user} ${entry.movie} ${entry.rating}`);
    }
  }

  const baseName = baseFileName || path.basename(testFilePath, ".txt");
  const validationPath = path.join(`${baseName}_validation.txt`);
  const testPath = path.join(`${baseName}_test.txt`);

  fs.writeFileSync(validationPath, validationLines.join("\n"));
  fs.writeFileSync(testPath, testLines2.join("\n"));

  console.log(`Split ${testFilePath} into validation and test sets.`);
  console.log(
    `- Validation: ${validationPath} (${validationUsers.length} users)`,
  );
  console.log(`- Test: ${testPath} (${testUsers2.length} users)`);

  return { validationPath, testPath };
}

/**
 * Load test data with both known ratings and ratings to predict
 * @param {Array<string>} testDataLines - Array of lines from the test data file.
 * @returns {Array<Object>} - Array of test user objects.
 */
function loadTestData(testDataLines) {
  const testUsers = [];
  let currentUser = null;
  let currentUserId = null;

  const isValidationFile =
    testDataLines[0] &&
    testDataLines[0].trim().split(" ").length === 3 &&
    !testDataLines.some((line) => line.trim().endsWith(" 0"));

  for (const line of testDataLines) {
    if (!line.trim()) continue;
    const [user, movie, rating] = line.split(" ").map(Number);
    if (currentUserId !== user) {
      if (currentUser) testUsers.push(currentUser);
      currentUserId = user;
      currentUser = {
        user,
        knownRatings: {},
        missing: [],
        actual: {},
      };
    }

    if (isValidationFile) {
      currentUser.actual[movie] = rating;
      currentUser.knownRatings[movie] = rating;
    } else if (rating === 0) {
      currentUser.missing.push(movie);
    } else {
      currentUser.knownRatings[movie] = rating;
    }
  }
  if (currentUser) testUsers.push(currentUser);
  return testUsers;
}

/**
 * Calculate Mean Absolute Error for predictions
 * @param {Object} predictions - Predicted ratings
 * @param {Object} actual - Actual ratings
 * @returns {number} - MAE value
 */
function calculateMAE(predictions, actual) {
  let totalError = 0;
  let count = 0;

  for (const movieId in actual) {
    if (predictions[movieId] !== undefined) {
      totalError += Math.abs(predictions[movieId] - actual[movieId]);
      count++;
    }
  }

  return count > 0 ? totalError / count : 0;
}

/**
 * Calculate Root Mean Square Error for predictions
 * @param {Object} predictions - Predicted ratings
 * @param {Object} actual - Actual ratings
 * @returns {number} - RMSE value
 */
function calculateRMSE(predictions, actual) {
  let totalSqError = 0;
  let count = 0;

  for (const movieId in actual) {
    if (predictions[movieId] !== undefined) {
      const error = predictions[movieId] - actual[movieId];
      totalSqError += error * error;
      count++;
    }
  }

  return count > 0 ? Math.sqrt(totalSqError / count) : 0;
}

/**
 * Create a validation test set by replacing some ratings with zeros
 * @param {Array<Object>} testUsers - Array of test user objects
 * @param {number} ratio - Ratio of ratings to hide (0-1)
 * @returns {Array<Object>} - Modified test users array
 */
function createValidationSet(testUsers, ratio = 0.5) {
  const validationUsers = [];

  for (const user of testUsers) {
    const validationUser = {
      user: user.user,
      knownRatings: {},
      missing: [],
      actual: {},
    };

    const movieIds = Object.keys(user.knownRatings);
    for (let i = movieIds.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [movieIds[i], movieIds[j]] = [movieIds[j], movieIds[i]];
    }

    const cutoff = Math.floor(movieIds.length * (1 - ratio));

    for (let i = 0; i < movieIds.length; i++) {
      const movieId = Number(movieIds[i]);
      const rating = user.knownRatings[movieId];

      if (i < cutoff) {
        validationUser.knownRatings[movieId] = rating;
      } else {
        validationUser.missing.push(movieId);
        validationUser.actual[movieId] = rating;
      }
    }

    validationUsers.push(validationUser);
  }

  return validationUsers;
}

/**
 * Evaluate an algorithm on the validation set
 * @param {object} recommender - Recommendation algorithm instance
 * @param {Array<Object>} validationUsers - Validation set users
 * @returns {Object} - Evaluation metrics
 */
function evaluateAlgorithm(recommender, validationUsers) {
  let totalMAE = 0;
  let totalRMSE = 0;
  let userCount = 0;
  let totalPredictions = 0;

  for (const user of validationUsers) {
    if (user.missing.length === 0) continue;

    const predictions = recommender.predict(
      user.user,
      user.knownRatings,
      user.missing,
    );

    const mae = calculateMAE(predictions, user.actual);
    const rmse = calculateRMSE(predictions, user.actual);

    if (mae > 0) {
      totalMAE += mae;
      totalRMSE += rmse;
      userCount++;
      totalPredictions += Object.keys(user.actual).length;
    }
  }

  return {
    averageMAE: userCount > 0 ? totalMAE / userCount : 0,
    averageRMSE: userCount > 0 ? totalRMSE / userCount : 0,
    userCount,
    totalPredictions,
  };
}

/**
 * Main function for the evaluation tool
 */
(async () => {
  const args = process.argv.slice(2);

  if (args.length < 1) {
    console.error("Usage: test.js <command> [options]");
    console.error("\nCommands:");
    console.error(
      "split <test_file> [base_name] - Split a test file into validation and test sets",
    );
    console.error(
      "evaluate <algorithm> <test_file> [validation_file] - Evaluate algorithm on test or validation set",
    );
    console.error(
      "compare <test_file> [validation_file] - Compare all algorithms on test or validation set",
    );
    process.exit(1);
  }

  const command = args[0];

  try {
    if (command === "split") {
      if (args.length < 2) {
        console.error("Usage: test.js split <test_file> [base_name]");
        console.error("\nOptions:");
        console.error("base_name - Optional base name for output files");
        process.exit(1);
      }

      const testFilePath = args[1];
      const baseFileName = args[2];
      await splitTestFile(testFilePath, baseFileName);
    } else if (command === "evaluate") {
      if (args.length < 3) {
        console.error(
          "Usage: test.js evaluate  <algorithm> <test_file> [validation_file]",
        );
        console.error("\nAvailable algorithms:");
        console.error("user-cosine - User-based CF with Cosine similarity");
        console.error("user-pearson - User-based CF with Pearson correlation");
        console.error("user-iuf - User-based CF with inverse user frequency");
        console.error("user-case - User-based CF with case modification");
        console.error("item - Item-based CF with Cosine similarity");
        console.error("hybrid - Custom hybrid recommendation algorithm");
        console.error("svd - SVD-based recommender");
        console.error("hybrid-svd - Hybrid recommendation algorithm with SVD");
        process.exit(1);
      }

      const algorithmType = args[1];
      const testFilePath = args[2];
      const validationFilePath = args.length > 3 ? args[3] : null;

      const trainDataLines = await readFile("./train.txt");
      const { userRatings, movieRatings } = loadTrainingData(trainDataLines);

      let validationUsers;

      if (validationFilePath) {
        console.log(`Using validation file: ${validationFilePath}`);
        const validationDataLines = await readFile(validationFilePath);

        validationUsers = loadTestData(validationDataLines);

        for (const user of validationUsers) {
          const movieIds = Object.keys(user.knownRatings);
          const halfPoint = Math.floor(movieIds.length / 2);

          for (let i = halfPoint; i < movieIds.length; i++) {
            const movieId = Number(movieIds[i]);
            user.actual[movieId] = user.knownRatings[movieId];
            user.missing.push(movieId);
            delete user.knownRatings[movieId];
          }
        }
      } else {
        console.log(`Generating validation set from: ${testFilePath}`);
        const testDataLines = await readFile(testFilePath);
        const testUsers = loadTestData(testDataLines);
        validationUsers = createValidationSet(testUsers);
      }

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

      console.log(`Evaluating ${algorithmType}...`);
      recommender.initialize(userRatings, movieRatings);

      const results = evaluateAlgorithm(recommender, validationUsers);

      console.log("\nResults:");
      console.log(
        `Mean Absolute Error (MAE): ${results.averageMAE.toFixed(4)}`,
      );
      console.log(
        `Root Mean Square Error (RMSE): ${results.averageRMSE.toFixed(4)}`,
      );
      console.log(`Users evaluated: ${results.userCount}`);
      console.log(`Total predictions: ${results.totalPredictions}`);

      if (results.userCount > 0) {
        const userErrors = [];
        for (const user of validationUsers) {
          if (user.missing.length === 0) continue;

          const predictions = recommender.predict(
            user.user,
            user.knownRatings,
            user.missing,
          );

          const mae = calculateMAE(predictions, user.actual);
          if (mae > 0) {
            userErrors.push({ user: user.user, mae });
          }
        }

        userErrors.sort((a, b) => a.mae - b.mae);
      }
    } else if (command === "compare") {
      if (args.length < 2) {
        console.error("Usage: test.js compare <test_file> [validation_file]");
        process.exit(1);
      }

      const testFilePath = args[1];
      const validationFilePath = args.length > 2 ? args[2] : null;

      const trainDataLines = await readFile("./train.txt");
      const { userRatings, movieRatings } = loadTrainingData(trainDataLines);

      let validationUsers;

      if (validationFilePath) {
        console.log(`Using validation file: ${validationFilePath}`);
        const validationDataLines = await readFile(validationFilePath);

        validationUsers = loadTestData(validationDataLines);

        for (const user of validationUsers) {
          const movieIds = Object.keys(user.knownRatings);
          const halfPoint = Math.floor(movieIds.length / 2);

          for (let i = halfPoint; i < movieIds.length; i++) {
            const movieId = Number(movieIds[i]);
            user.actual[movieId] = user.knownRatings[movieId];
            user.missing.push(movieId);
            delete user.knownRatings[movieId];
          }
        }
      } else {
        console.log(`Generating validation set from: ${testFilePath}`);
        const testDataLines = await readFile(testFilePath);
        const testUsers = loadTestData(testDataLines);
        validationUsers = createValidationSet(testUsers);
      }

      const algorithms = [
        { name: "user-cosine", recommender: new UserBasedCF("cosine") },
        { name: "user-pearson", recommender: new UserBasedCF("pearson") },
        { name: "user-iuf", recommender: new ExtendedUserBasedCF("iuf") },
        { name: "user-case", recommender: new ExtendedUserBasedCF("case") },
        { name: "item", recommender: new ItemBasedCF() },
        { name: "hybrid", recommender: new HybridCF() },
        { name: "svd", recommender: new SVDRecommender() },
        { name: "hybrid-svd", recommender: new HybridSVDCF() },
      ];

      console.log("Comparing all algorithms...");
      console.log("This may take some time...\n");

      const results = [];

      for (const algo of algorithms) {
        console.log(`Evaluating ${algo.name}...`);
        algo.recommender.initialize(userRatings, movieRatings);
        const metrics = evaluateAlgorithm(algo.recommender, validationUsers);
        results.push({
          name: algo.name,
          mae: metrics.averageMAE,
          rmse: metrics.averageRMSE,
        });
      }

      results.sort((a, b) => a.mae - b.mae);

      console.log("\nResults (sorted by MAE):");
      console.log("=========================");
      console.log("Algorithm        MAE      RMSE");
      console.log("-------------------------");

      for (const result of results) {
        const name = result.name.padEnd(16);
        console.log(
          `${name}${result.mae.toFixed(4)}  ${result.rmse.toFixed(4)}`,
        );
      }
    } else {
      console.error(`Unknown command: ${command}`);
      process.exit(1);
    }
  } catch (error) {
    console.error("Error:", error.message);
    process.exit(1);
  }
})();
