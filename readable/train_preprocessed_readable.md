# train_preprocessed.csv 可讀化摘要

- 總列數: 6956
- 欄位數: 19
- 原始檔: preprocessed/train_preprocessed.csv

## 欄位清單

index, ID, tweet, tweet_clean, tweet_raw_len, tweet_clean_len, split, ineffective, unnecessary, pharma, rushed, side-effect, mandatory, country, ingredients, political, none, conspiracy, religious

## 標籤分布

| label | count | ratio |
| --- | --- | --- |
| side-effect | 2025 | 29.11% |
| multi-label | 1359 | 19.54% |
| ineffective | 847 | 12.18% |
| rushed | 548 | 7.88% |
| pharma | 520 | 7.48% |
| none | 440 | 6.33% |
| mandatory | 367 | 5.28% |
| unnecessary | 255 | 3.67% |
| political | 239 | 3.44% |
| ingredients | 139 | 2.00% |
| conspiracy | 105 | 1.51% |
| country | 88 | 1.27% |
| religious | 24 | 0.35% |

## 前 20 筆預覽

| index | ID | split | label | tweet_clean | tweet_raw_len | tweet_clean_len |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1311981051720409089 | train | ineffective | [USER] They have no idea, they cant control the Flu with vaccine so what makes them think they ca... | 41 | 41 |
| 1 | 1361403925845401601 | train | unnecessary | [USER] Nvm I’ve had covid I’ve got enough antibodies to see me through you can keep your judgey v... | 19 | 19 |
| 2 | 1293488278361055233 | train | pharma | Coronavirus updates: Government partners with Moderna to make 100M vaccine doses [URL] Why are we... | 43 | 43 |
| 3 | 1305252218526990338 | train | rushed | [USER] U.K. Glaxo Smith Klein whistleblower who has found antigens in the upcoming Covid vaccine ... | 25 | 25 |
| 4 | 1376135683400687618 | train | ineffective,pharma | 3/ horse" AstraZeneca, not so much for the reduced effectiveness of the vaccine, if compared to o... | 47 | 47 |
| 5 | 1351207409071640578 | train | side-effect | After a video of his mother’s condition went viral, the son of a woman who was hospitalized after... | 40 | 40 |
| 6 | 1354543537078095880 | train | ineffective,rushed | [USER] [USER] One dose of the least efficacious vaccine to the most vulnerable in most cases. The... | 53 | 53 |
| 7 | 1335249891807932417 | train | mandatory | Please read through [USER] ’s thread on the COVID vaccine being pushed to front line workers in A... | 28 | 29 |
| 8 | 1298704879150653440 | train | country | [USER] I wouldnt take it if the vaccine is russian. I cant trust those homophobics and their hist... | 26 | 26 |
| 9 | 1364590447214145537 | train | ineffective | We're told that we have to be overly cautious, in case the vaccine does not offer enough protecti... | 45 | 46 |
| 10 | 1407341885052669960 | train | side-effect | Sen. Johnson: Democrat media suppress information about more than 5,000 vaccine deaths - [URL] | 14 | 14 |
| 11 | 1354844941168619524 | train | ingredients | [USER] [USER] About the disputed efficacy of the AZ vaccine, see here [URL] About the derivation ... | 36 | 36 |
| 12 | 1410873702523150336 | train | side-effect | [USER] Meanwhile the clots say "The final details of that will be released within the coming week... | 36 | 36 |
| 13 | 1385549673214132230 | train | rushed | [USER] [USER] [USER] [USER] ‘People’ ,not kids! Surely you protect your kids from things that mig... | 50 | 50 |
| 14 | 1376993890377547779 | train | unnecessary,rushed,side-effect | [USER] [USER] You assume no long term effects from this experimental vaccine. And if the risk of ... | 57 | 57 |
| 15 | 1349576125454852097 | train | unnecessary,pharma | [USER] [USER] can keep his vaccine syringe I’ll take covid with its “ non existent “ symptoms ove... | 31 | 32 |
| 16 | 1280532929790259208 | train | pharma | And taxpayers just gave $1.6 billion to Novavax, a company that has brought nothing to market, th... | 30 | 30 |
| 17 | 1374399606121844741 | train | rushed | [USER] Good grief. Between the flawed trials we heard about earlier, and now this, AstraZeneca is... | 46 | 46 |
| 18 | 1337572148584570880 | train | side-effect,ingredients | [USER] Dr. Mikovits warned Months ago that HIV had been added to the CV vaccs. Now, after taking ... | 36 | 36 |
| 19 | 1340312194215538688 | train | side-effect | The FDA is investigating allergic reactions to the Pfizer vaccine reported in multiple states. I ... | 36 | 36 |
