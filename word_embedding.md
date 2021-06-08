---
title: "Understanding Word Embedding"
author: "Arjun Paudel"
date: "6/7/2021"
output: 
  html_document: 
    keep_md: yes
editor_options: 
  chunk_output_type: console
---




```r
library(tidyverse)
library(tidytext)
library(textdata)
library(widyr)
```


```r
dt <- tibble(id = c(1,2,1,2),
             text = c("This is a good dog",
                      "Who is a good boy",
                      "Boy, it sure is hot today, huh?",
                      "This is a hot summer month")
             )
dt <- dt %>% arrange(id)
dt
```

```
## # A tibble: 4 x 2
##      id text                           
##   <dbl> <chr>                          
## 1     1 This is a good dog             
## 2     1 Boy, it sure is hot today, huh?
## 3     2 Who is a good boy              
## 4     2 This is a hot summer month
```


```r
tidy_dt <- dt %>% 
  unnest_tokens(word, text)
tidy_dt
```

```
## # A tibble: 23 x 2
##       id word 
##    <dbl> <chr>
##  1     1 this 
##  2     1 is   
##  3     1 a    
##  4     1 good 
##  5     1 dog  
##  6     1 boy  
##  7     1 it   
##  8     1 sure 
##  9     1 is   
## 10     1 hot  
## # ... with 13 more rows
```


```r
tidy_ngram_dt <- dt %>% 
  unnest_tokens(words, text, token = "ngrams", n = 2, collapse = "id") %>% 
  mutate(.ngramID = row_number()) %>% 
  unnest_tokens(word, words) %>% 
  unite(.ngramID, id, .ngramID)

tidy_pmi <- tidy_ngram_dt %>% 
  pairwise_pmi(item = word, feature = .ngramID, sort = TRUE)
tidy_pmi
```

```
## # A tibble: 34 x 3
##    item1  item2    pmi
##    <chr>  <chr>  <dbl>
##  1 huh    today  2.35 
##  2 today  huh    2.35 
##  3 month  summer 2.35 
##  4 summer month  2.35 
##  5 sure   it     1.66 
##  6 it     sure   1.66 
##  7 who    is     0.965
##  8 dog    good   0.965
##  9 good   dog    0.965
## 10 boy    dog    0.965
## # ... with 24 more rows
```

> Notice the use of `collapse = "id"`
>
> Without this, the bigram tokenizer will stop at the end of each row and will not crossover to next row.
>
> With `collapse = "id"` it will cross over to next row with same id and treat last word of previous row and first word of next row together.
>
> In our case, row1 and row3 have same ***id***, so we can think of this as one single long sentence that we just chose to represent in two rows. `collapse = "id"` will treat it such.
>
> Sometimes we read a big text file, certain number of lines at a time. So all the text of same document can end up in different rows. They will share the same document id but not row number. Using `collapse = "document_id"` will treat all those different rows as part of same document.

Implementation from [smltar](https://smltar.com/embeddings.html)


```r
nested_words <- tidy_dt %>%
  nest(words = c(word))

nested_words
```

```
## # A tibble: 2 x 2
##      id words            
##   <dbl> <list>           
## 1     1 <tibble [12 x 1]>
## 2     2 <tibble [11 x 1]>
```

```r
slide_windows <- function(tbl, window_size) {
  skipgrams <- slider::slide(
    tbl, 
    ~.x, 
    .after = window_size - 1, 
    .step = 1, 
    .complete = TRUE
  )
  
  safe_mutate <- safely(mutate)
  
  out <- map2(skipgrams,
              1:length(skipgrams),
              ~ safe_mutate(.x, window_id = .y))

  out %>%
    transpose() %>%
    pluck("result") %>%
    compact() %>%
    bind_rows()
}


tidy_pmi2 <- nested_words %>%
  mutate(words = map(words, slide_windows, 2L)) %>%
  unnest(words) %>%
  unite(window_id, id, window_id) %>%
  pairwise_pmi(word, window_id, sort = TRUE)

tidy_pmi2
```

```
## # A tibble: 34 x 3
##    item1  item2    pmi
##    <chr>  <chr>  <dbl>
##  1 huh    today  2.35 
##  2 today  huh    2.35 
##  3 month  summer 2.35 
##  4 summer month  2.35 
##  5 sure   it     1.66 
##  6 it     sure   1.66 
##  7 who    is     0.965
##  8 dog    good   0.965
##  9 good   dog    0.965
## 10 boy    dog    0.965
## # ... with 24 more rows
```

PMI calculated by both methods is identical.


```r
identical(tidy_pmi, tidy_pmi2)
```

```
## [1] TRUE
```

### Represent words as numeric vectors

Create 5 dimensional vector using `widely_svd()` from **widyr** package


```r
set.seed(13)
tidy_word_vectors <- tidy_pmi %>% 
  widely_svd(
    item1, item2, pmi,
    nv = 5, maxit = 10000
  )
```

Using `irlba()` from **irlba** library


```r
pmi_matrix <- tidy_pmi %>% 
  cast_sparse(item1, item2, pmi)

class(pmi_matrix)
```

```
## [1] "dgCMatrix"
## attr(,"package")
## [1] "Matrix"
```

```r
library(irlba)
```

```
## Loading required package: Matrix
```

```
## 
## Attaching package: 'Matrix'
```

```
## The following objects are masked from 'package:tidyr':
## 
##     expand, pack, unpack
```

```r
set.seed(13)
pmi_svd <- irlba(pmi_matrix, nv=5, maxit = 10000)
pmi_svd
```

```
## $d
## [1] 2.731022 2.729388 2.351375 2.134447 2.031151
## 
## $u
##               [,1]          [,2]          [,3]        [,4]         [,5]
##  [1,]  0.427830391 -0.4286980821  6.603296e-01 -0.02671928  0.008351493
##  [2,]  0.496906697  0.4976166154  2.529126e-01 -0.02425427 -0.007214136
##  [3,]  0.427830391 -0.4286980821 -6.603296e-01 -0.02671928  0.008351493
##  [4,]  0.496906697  0.4976166154 -2.529126e-01 -0.02425427 -0.007214136
##  [5,] -0.015612240  0.0103428563  1.432703e-16 -0.47005832  0.542474331
##  [6,] -0.013734491 -0.0060186571 -8.882720e-16 -0.58136797 -0.655402784
##  [7,] -0.025811206  0.0237291983 -1.586227e-15 -0.06530127  0.026278443
##  [8,] -0.007579327 -0.0006898637 -6.014546e-16 -0.31800405 -0.239164495
##  [9,] -0.009407261  0.0027007791  2.556512e-15 -0.22519094  0.056061119
## [10,] -0.012041000 -0.0007497449  2.543678e-15 -0.47813140  0.447294724
## [11,]  0.363775941 -0.3628283934 -8.227170e-16  0.01145767 -0.005164850
## [12,] -0.073041509 -0.0671095958  7.300637e-16 -0.14442533 -0.055306737
## [13,] -0.017434323  0.0139134335 -2.400901e-15 -0.16322377 -0.107999213
## [14,] -0.026987119 -0.0116183619 -4.007035e-16 -0.07815816 -0.008380762
## 
## $v
##               [,1]          [,2]          [,3]        [,4]         [,5]
##  [1,]  0.496906697 -0.4976166154  6.603296e-01 -0.02425427  0.007214136
##  [2,]  0.427830391  0.4286980821  2.529126e-01 -0.02671928 -0.008351493
##  [3,]  0.496906697 -0.4976166154 -6.603296e-01 -0.02425427  0.007214136
##  [4,]  0.427830391  0.4286980821 -2.529126e-01 -0.02671928 -0.008351493
##  [5,] -0.013734491  0.0060186571 -1.388354e-17 -0.58136797  0.655402784
##  [6,] -0.015612240 -0.0103428563  1.524476e-15 -0.47005832 -0.542474331
##  [7,] -0.073041509  0.0671095958 -3.910444e-15 -0.14442533  0.055306737
##  [8,] -0.009407261 -0.0027007791  3.667094e-15 -0.22519094 -0.056061119
##  [9,] -0.007579327  0.0006898637 -7.394943e-15 -0.31800405  0.239164495
## [10,] -0.012041000  0.0007497449 -5.364881e-15 -0.47813140 -0.447294724
## [11,]  0.363775941  0.3628283934  1.447619e-14  0.01145767  0.005164850
## [12,] -0.025811206 -0.0237291983 -1.175339e-14 -0.06530127 -0.026278443
## [13,] -0.017434323 -0.0139134335  1.988635e-14 -0.16322377  0.107999213
## [14,] -0.026987119  0.0116183619  2.561970e-14 -0.07815816  0.008380762
## 
## $iter
## [1] 2
## 
## $mprod
## [1] 38
```

```r
word_vectors <- pmi_svd$u
rownames(word_vectors) <- rownames(pmi_matrix)
```

Values are almost identical except for third dimension. Not sure if this is rounding error or if these two libraries differ how they implement SVD.  There are also some sign inconsistencies.

> Actually looking at source code of `widely_svd()`, it calls `irba()`. Not really sure what is causing the discrepancy.

> Realized that the inconistencies were due to random nature of the process. Both svd processes resulted in identical result after setting same seed before function calls.


```r
word_vectors
```

```
##                [,1]          [,2]          [,3]        [,4]         [,5]
## huh     0.427830391 -0.4286980821  6.603296e-01 -0.02671928  0.008351493
## today   0.496906697  0.4976166154  2.529126e-01 -0.02425427 -0.007214136
## month   0.427830391 -0.4286980821 -6.603296e-01 -0.02671928  0.008351493
## summer  0.496906697  0.4976166154 -2.529126e-01 -0.02425427 -0.007214136
## sure   -0.015612240  0.0103428563  1.432703e-16 -0.47005832  0.542474331
## it     -0.013734491 -0.0060186571 -8.882720e-16 -0.58136797 -0.655402784
## who    -0.025811206  0.0237291983 -1.586227e-15 -0.06530127  0.026278443
## dog    -0.007579327 -0.0006898637 -6.014546e-16 -0.31800405 -0.239164495
## good   -0.009407261  0.0027007791  2.556512e-15 -0.22519094  0.056061119
## boy    -0.012041000 -0.0007497449  2.543678e-15 -0.47813140  0.447294724
## hot     0.363775941 -0.3628283934 -8.227170e-16  0.01145767 -0.005164850
## is     -0.073041509 -0.0671095958  7.300637e-16 -0.14442533 -0.055306737
## this   -0.017434323  0.0139134335 -2.400901e-15 -0.16322377 -0.107999213
## a      -0.026987119 -0.0116183619 -4.007035e-16 -0.07815816 -0.008380762
```

```r
(tidy_word_vectors %>% cast_sparse(item1, dimension, value))
```

```
## 14 x 5 sparse Matrix of class "dgCMatrix"
##                   1             2             3           4            5
## huh     0.427830391 -0.4286980821  6.603296e-01 -0.02671928  0.008351493
## today   0.496906697  0.4976166154  2.529126e-01 -0.02425427 -0.007214136
## month   0.427830391 -0.4286980821 -6.603296e-01 -0.02671928  0.008351493
## summer  0.496906697  0.4976166154 -2.529126e-01 -0.02425427 -0.007214136
## sure   -0.015612240  0.0103428563  1.432703e-16 -0.47005832  0.542474331
## it     -0.013734491 -0.0060186571 -8.882720e-16 -0.58136797 -0.655402784
## who    -0.025811206  0.0237291983 -1.586227e-15 -0.06530127  0.026278443
## dog    -0.007579327 -0.0006898637 -6.014546e-16 -0.31800405 -0.239164495
## good   -0.009407261  0.0027007791  2.556512e-15 -0.22519094  0.056061119
## boy    -0.012041000 -0.0007497449  2.543678e-15 -0.47813140  0.447294724
## hot     0.363775941 -0.3628283934 -8.227170e-16  0.01145767 -0.005164850
## is     -0.073041509 -0.0671095958  7.300637e-16 -0.14442533 -0.055306737
## this   -0.017434323  0.0139134335 -2.400901e-15 -0.16322377 -0.107999213
## a      -0.026987119 -0.0116183619 -4.007035e-16 -0.07815816 -0.008380762
```

#### Similarity

Similarity between vector representation of words can be found using cosine similarity.        
$sim(x,y) = \frac{x.y}{||x||.||y||}$

$x = (5, 0, 3, 0, 2, 0, 0, 2, 0, 0)$ and  $y = (3, 0, 2, 0, 1, 1, 0, 1, 0, 1)$     
$x.y = x^{t}.y = 5*3+0*0+3*2+0*0+2*1+0*1+0*0+0*1 = 25$          
$||x|| = \sqrt{5^2+0^2+3^2+0^2+2^2+0^2+0^2+2^2+0^2+0^2} = 6.48$         
$||y|| = \sqrt{3^2+0^2+2^2+0^2+1^2+1^2+0^2+1^2+0^2+1^2} = 4.12$            
$sim(x,y) = \frac{25}{6.48*4.12} = 0.94$        


**Some matrix multiplication quirks to understand.**

Let's create a 2x2 matrix.

```r
a <- matrix(c(1,2,1,2), nrow = 2, byrow = TRUE)
a
```

```
##      [,1] [,2]
## [1,]    1    2
## [2,]    1    2
```

If we slice one row or one column, the result is just a vector not a matrix.

```r
a[1,]
```

```
## [1] 1 2
```

```r
a[,1]
```

```
## [1] 1 1
```

Matrix have dimensions but the dimension of a vector is `NULL`

```r
a %>% dim()
```

```
## [1] 2 2
```

```r
a[1,] %>% dim()
```

```
## NULL
```

```r
a[,1] %>% dim()
```

```
## NULL
```

Since the dimension of an vector is NULL it can act as a row vector or column vector.    
In this case it acts as a 1x2 row vector and so can be multiplied by a 2x2 matrix.    
$a^*_{1\times2}*a_{2\times2}$

```r
a[1,]%*%a
```

```
##      [,1] [,2]
## [1,]    3    6
```

Here it acts as 2x1 column vector and can multiply a.         
$a_{2\times2}*a^*_{2\times1}$

```r
a%*%a[1,]
```

```
##      [,1]
## [1,]    5
## [2,]    5
```

Outcome of both above operations result in a matrix of appropriate dimensions.


```r
nearest_neighbors <- function(df, token) {
  df %>%
    widely(
      ~ {
        y <- (. %*% (.[token, ]))[,1]
        res <- y / 
          (sqrt(rowSums(. ^ 2)) * sqrt(sum(.[token, ] ^ 2)))
      matrix(res, ncol = 1, dimnames = list(x = names(res)))
      },
      sort = TRUE
    )(item1, dimension, value) %>% 
  select(-item2)
}
```

Implementation from the book

```r
nearest_neighbors2 <- function(df, token) {
  df %>%
    widely(
      ~ {
        y <- .[rep(token, nrow(.)), ]
        res <- rowSums(. * y) / 
          (sqrt(rowSums(. ^ 2)) * sqrt(sum(.[token, ] ^ 2)))

        matrix(res, ncol = 1, dimnames = list(x = names(res)))
        },
      sort = TRUE
    )(item1, dimension, value) %>%
    select(-item2)
}
```

Identical results

```r
tidy_word_vectors %>% nearest_neighbors2("today")
```

```
## # A tibble: 14 x 2
##    item1       value
##    <chr>       <dbl>
##  1 today   1        
##  2 summer  0.771    
##  3 huh     0.249    
##  4 this    0.0204   
##  5 dog     0.0179   
##  6 it      0.0138   
##  7 good    0.00994  
##  8 sure    0.00908  
##  9 who     0.00640  
## 10 boy     0.00411  
## 11 hot    -0.0000712
## 12 month  -0.249    
## 13 a      -0.275    
## 14 is     -0.479
```

```r
tidy_word_vectors %>% nearest_neighbors("today")
```

```
## # A tibble: 14 x 2
##    item1       value
##    <chr>       <dbl>
##  1 today   1        
##  2 summer  0.771    
##  3 huh     0.249    
##  4 this    0.0204   
##  5 dog     0.0179   
##  6 it      0.0138   
##  7 good    0.00994  
##  8 sure    0.00908  
##  9 who     0.00640  
## 10 boy     0.00411  
## 11 hot    -0.0000712
## 12 month  -0.249    
## 13 a      -0.275    
## 14 is     -0.479
```

#### Word embedding

Now we have documents made up of multiple words and we know the vector representation of those words. So how do we go about representing the documents in vector form.    
One common approach is to aggregate the word vectors of all the words that make up the document. We could sum all those word vectors of a document or average them or take the min/max across each dimension.    

We need a matrix of *document_id*, *words* and the count of how many times that word appears in that document.

Document-term matrix

```r
tidy_dt %>% head()
```

```
## # A tibble: 6 x 2
##      id word 
##   <dbl> <chr>
## 1     1 this 
## 2     1 is   
## 3     1 a    
## 4     1 good 
## 5     1 dog  
## 6     1 boy
```

```r
doc_term_matrix <- tidy_dt %>% 
  count(id, word) %>% 
  cast_sparse(id, word, n)
```

Word embedding matrix

```r
tidy_word_vectors %>% head()
```

```
## # A tibble: 6 x 3
##   item1  dimension   value
##   <chr>      <int>   <dbl>
## 1 huh            1  0.428 
## 2 today          1  0.497 
## 3 month          1  0.428 
## 4 summer         1  0.497 
## 5 sure           1 -0.0156
## 6 it             1 -0.0137
```

```r
embedding_matrix <- tidy_word_vectors %>%
  cast_sparse(item1, dimension, value)
embedding_matrix
```

```
## 14 x 5 sparse Matrix of class "dgCMatrix"
##                   1             2             3           4            5
## huh     0.427830391 -0.4286980821  6.603296e-01 -0.02671928  0.008351493
## today   0.496906697  0.4976166154  2.529126e-01 -0.02425427 -0.007214136
## month   0.427830391 -0.4286980821 -6.603296e-01 -0.02671928  0.008351493
## summer  0.496906697  0.4976166154 -2.529126e-01 -0.02425427 -0.007214136
## sure   -0.015612240  0.0103428563  1.432703e-16 -0.47005832  0.542474331
## it     -0.013734491 -0.0060186571 -8.882720e-16 -0.58136797 -0.655402784
## who    -0.025811206  0.0237291983 -1.586227e-15 -0.06530127  0.026278443
## dog    -0.007579327 -0.0006898637 -6.014546e-16 -0.31800405 -0.239164495
## good   -0.009407261  0.0027007791  2.556512e-15 -0.22519094  0.056061119
## boy    -0.012041000 -0.0007497449  2.543678e-15 -0.47813140  0.447294724
## hot     0.363775941 -0.3628283934 -8.227170e-16  0.01145767 -0.005164850
## is     -0.073041509 -0.0671095958  7.300637e-16 -0.14442533 -0.055306737
## this   -0.017434323  0.0139134335 -2.400901e-15 -0.16322377 -0.107999213
## a      -0.026987119 -0.0116183619 -4.007035e-16 -0.07815816 -0.008380762
```



```r
doc_term_matrix %*% embedding_matrix 
```

```
## 2 x 5 sparse Matrix of class "dgCMatrix"
##          1          2             3         4         5
## 1 2.103253 -0.1719476 -1.074104e-15 -2.293845 0.2009296
## 2 1.652736  0.1300741  1.320659e+00 -1.566547 0.8729139
```


```r
test_dtm <- matrix(c(1,2,0,2,0,1), nrow = 2)
dimnames(test_dtm) <- list(c(1,2), c("the", "good", "boy"))
test_dtm
```

```
##   the good boy
## 1   1    0   0
## 2   2    2   1
```

```r
test_embed <- matrix(c(.2,.4,.1,1,2,3,.5,.25,.4, .02,.15,.8), nrow = 3)
dimnames(test_embed) <- list(c("the", "good", "boy"), c("d1", "d2", "d3", "d4"))
test_embed
```

```
##       d1 d2   d3   d4
## the  0.2  1 0.50 0.02
## good 0.4  2 0.25 0.15
## boy  0.1  3 0.40 0.80
```

```r
doc_matrix <- test_dtm %*% test_embed
doc_matrix
```

```
##    d1 d2  d3   d4
## 1 0.2  1 0.5 0.02
## 2 1.3  9 1.9 1.14
```

We can replicate this using simple count, sum and multiplication in tidy format.


```r
tidy_embed<-test_embed %>% as_tibble(rownames = "word") %>% 
  pivot_longer(!word, names_to = "dimension", values_to = "value")

tidy_dtm <- tibble(word = rep(c("the", "good", "boy"), times = 2),
                   id = rep(c(1,2), each = 3,),
                   n = c(1,0,0,2,2,1)
)

tidy_doc_matrix <- tidy_dtm %>% left_join(tidy_embed, by = "word") %>% 
  mutate(value = n*value) %>% 
  pivot_wider(names_from = dimension, values_from = value,values_fill = 0) %>% 
  group_by(id) %>% 
  summarise(across (starts_with("d"), sum))
```

Both results are identical


```r
doc_matrix
```

```
##    d1 d2  d3   d4
## 1 0.2  1 0.5 0.02
## 2 1.3  9 1.9 1.14
```

```r
tidy_doc_matrix
```

```
## # A tibble: 2 x 5
##      id    d1    d2    d3    d4
##   <dbl> <dbl> <dbl> <dbl> <dbl>
## 1     1   0.2     1   0.5  0.02
## 2     2   1.3     9   1.9  1.14
```

