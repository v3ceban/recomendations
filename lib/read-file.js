import fs from "fs";
import readline from "readline";

/**
 * Read a file line by line and return an array of lines.
 * @param {string} filePath - Path to the training data file.
 * @returns {Promise<Array<string>>} - Array of lines in the file.
 */
export async function readFile(filePath) {
  const fileStream = fs.createReadStream(filePath);
  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity,
  });

  const lines = [];
  for await (const line of rl) {
    lines.push(line);
  }

  return lines;
}
