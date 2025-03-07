## Classes

<dl>
<dt><a href="#HybridSVDCF">HybridSVDCF</a></dt>
<dd><p>Enhanced Hybrid Collaborative Filtering algorithm.
Combines multiple recommenders with adaptive weighting and normalization.</p>
</dd>
<dt><a href="#HybridCF">HybridCF</a></dt>
<dd><p>Hybrid Collaborative Filtering algorithm.
A custom algorithm that combines user-based and item-based approaches
with additional features for improved recommendation accuracy.</p>
</dd>
<dt><a href="#ItemBasedCF">ItemBasedCF</a></dt>
<dd><p>Item-Based Collaborative Filtering recommendation algorithm.
Uses cosine similarity between items (movies) instead of users.</p>
</dd>
<dt><a href="#SVDRecommender">SVDRecommender</a></dt>
<dd><p>Matrix Factorization based recommender using Singular Value Decomposition (SVD)
This is an implementation of a simplified SVD approach for collaborative filtering.</p>
</dd>
<dt><a href="#SVDRecommender">SVDRecommender</a></dt>
<dd><p>Matrix Factorization based recommender using Singular Value Decomposition (SVD)
This is an implementation of a simplified SVD approach for collaborative filtering.</p>
</dd>
<dt><a href="#ExtendedUserBasedCF">ExtendedUserBasedCF</a></dt>
<dd><p>Extended User-Based Collaborative Filtering with advanced features:</p>
<ol>
<li>Inverse User Frequency - Weights rare item ratings more strongly</li>
<li>Case Modification - Adjusts similarity based on number of common items</li>
</ol>
</dd>
<dt><a href="#UserBasedCF">UserBasedCF</a></dt>
<dd><p>User-Based Collaborative Filtering recommendation algorithm.
Implements both Cosine similarity and Pearson correlation methods.</p>
</dd>
</dl>

## Functions

<dl>
<dt><a href="#readFile">readFile(filePath)</a> ⇒ <code>Promise.&lt;Array.&lt;string&gt;&gt;</code></dt>
<dd><p>Read a file line by line and return an array of lines.</p>
</dd>
<dt><a href="#loadTrainingData">loadTrainingData(trainLines)</a> ⇒ <code>Promise.&lt;Object&gt;</code></dt>
<dd><p>Parse training data lines into a structured format.</p>
</dd>
<dt><a href="#loadTestData">loadTestData(testDataLines)</a> ⇒ <code>Promise.&lt;Array.&lt;Object&gt;&gt;</code></dt>
<dd><p>Group test data lines by user.
In the test file, ratings with value 0 indicate missing values that need prediction.
The output is an array of objects for each test user with known ratings and movies to predict.</p>
</dd>
<dt><a href="#printPredictedData">printPredictedData(testDataLines, predictions, userId)</a></dt>
<dd><p>Print test data with predictions replacing zeros</p>
</dd>
</dl>

<a name="readFile"></a>

## readFile(filePath) ⇒ <code>Promise.&lt;Array.&lt;string&gt;&gt;</code>
Read a file line by line and return an array of lines.

**Kind**: global function  
**Returns**: <code>Promise.&lt;Array.&lt;string&gt;&gt;</code> - - Array of lines in the file.  

| Param | Type | Description |
| --- | --- | --- |
| filePath | <code>string</code> | Path to the training data file. |

<a name="loadTrainingData"></a>

## loadTrainingData(trainLines) ⇒ <code>Promise.&lt;Object&gt;</code>
Parse training data lines into a structured format.

**Kind**: global function  
**Returns**: <code>Promise.&lt;Object&gt;</code> - - Parsed user rating data and movie rating data.  

| Param | Type | Description |
| --- | --- | --- |
| trainLines | <code>Array.&lt;string&gt;</code> | Array of lines from the training data file. |

<a name="loadTestData"></a>

## loadTestData(testDataLines) ⇒ <code>Promise.&lt;Array.&lt;Object&gt;&gt;</code>
Group test data lines by user.
In the test file, ratings with value 0 indicate missing values that need prediction.
The output is an array of objects for each test user with known ratings and movies to predict.

**Kind**: global function  
**Returns**: <code>Promise.&lt;Array.&lt;Object&gt;&gt;</code> - - Array of test user objects.
Each object contains:
- user: user id
- knownRatings: object mapping movie id to rating
- missing: array of movie ids that require prediction  

| Param | Type | Description |
| --- | --- | --- |
| testDataLines | <code>Array.&lt;string&gt;</code> | Array of lines from the test data file. |

<a name="printPredictedData"></a>

## printPredictedData(testDataLines, predictions, userId)
Print test data with predictions replacing zeros

**Kind**: global function  

| Param | Type | Description |
| --- | --- | --- |
| testDataLines | <code>Array.&lt;string&gt;</code> | Array of lines from the test data file. |
| predictions | <code>Object</code> | Object containing predicted ratings |
| userId | <code>number</code> | ID of the user whose predictions to print |

