# 알고리즘 공부 

### 알고리즘 개념
1. 알고리즘이란?
  > 주어진 입력을 출력으로 이끄는 잘 정의된 계산 절차   
  > 계산 문제를 정의하기 위해 입력과 출력의 관계를 잘 정의해야 하는데, 이러한 관계를 구현할 수 있는 계산과정  

2. 타당한 알고리즘
  > 알고리즘이 모든 입력 사례(Instance)에 대해 항상 올바른 출력을 내고, 종료할 경우를 타당하다고 한다  
  > 일부 입력 사례에 종료하지 않거나, 잘못된 답을 도출할 경우 타당하지 않다 일컺는다  

3. 어떤 문제를 알고리즘으로 풀어야할까?
  > 문제들은 공통적으로 "후보는 많지만, 대부분이 문제의 답이 아니다. 최상의 답을 찾는 것을 도전한다", "실용적인 사례를 주변에서 찾을 수 있다"와 같은 특징을 갖는다.
  
4. 알고리즘은 왜 필요할까?
  > 컴퓨터는 유한한 성능을 갖는다. 이로인해 공간적, 시간적 제약이 생긴다.  
  > 이러한 제약사항을 고려하여, 자원을 보다 효율적으로 사용해야 한다. = 효율적인 알고리즘이 필요하다.
  
5. 공부방법
  > 입력과 출력 그리고 사례(인스턴스)를 확실하게 이해한다  
  > 사례를 통해 아이디어(기본적인 이해)를 정리한다  
  > 아이디어를 유사코드로 만들고, 성능을 분석-파악한다  
  > 시간, 공간 복잡도를 이용하여 알고리즘을 효율적으로 최적화한다.  
  
### 삽입정렬
  - 입력 : n개 수들의 수열  (A Sequence of n numbers <a1, a2, ... , an>)
  - 출력 : a1 ≤ a2 ≤ a3 ≤ ... ≤ an을 만족하는 입력 수열의 순열(재배치)   (A Permutation <b1, b2, ... , bn> of the input sequence such that b1 ≤ b2 ≤ ... ≤ bn)
  - 유사코드
  ~~~
  for j = 2 to A.length
      key = A[j]
      i = j-1
      while i>0 and A[i] > key
          A[i+1] = A[i]
          i = i-1
      A[i+1] = key
  ~~~
  - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/%EC%82%BD%EC%9E%85%EC%A0%95%EB%A0%AC.gif" width="500" height="300" />  
  
  - 성능 분석하기
  ~~~                                     cost          times
   for j = 2 to A.length                  c₁            n
      key = A[j]                          c₂            n-1
      i = j-1                             c₃            n-1
      while i>0 and A[i] > key            c₄            Σtᴊ (j=2 ~ n)
          A[i+1] = A[i]                   c₅            Σ(tᴊ-1) (j=2 ~ n)
          i = i-1                         c₆            Σ(tᴊ-1) (j=2 ~ n)
      A[i+1] = key                        c₇            n-1
      
      *) tᴊ는 j가 2, 3, .., n 일때, while 루프의 검사가 실행되는 횟수를 의미
  ~~~
    > 수행시간은 각 명령문 수행시간의 합  
    > T(n) = c₁n + c₂(n-1) + c₃(n-1) + c₄Σtᴊ (j=2 ~ n) + c₅Σ(tᴊ-1) (j=2 ~ n) + c₆Σ(tᴊ-1) (j=2 ~ n) + c₇(n-1)  
    > (최악의 경우) T(n) = c₁n + c₂(n-1) + c₃(n-1) + c₄(n(n+1)/2-1) + c₅(n(n-1)/2)+ c₆(n(n-1)/2) + c₇(n-1)  
    > (최악의 경우) T(n) = an² + bn + c
 <hr/>
 
 
### 합병정렬
  - 입력 : n개 수들의 수열  (A Sequence of n numbers <a1, a2, ... , an>)
  - 출력 : a1 ≤ a2 ≤ a3 ≤ ... ≤ an을 만족하는 입력 수열의 순열(재배치)   (A Permutation <b1, b2, ... , bn> of the input sequence such that b1 ≤ b2 ≤ ... ≤ bn)
  - 분할(정렬할 n개 원소의 배열을 n/2개씩 부분 수열 두개로 분할),   
    정복(합병정렬을 통해 두 부분배열을 재귀적으로 정렬),   
    결합(정렬된 두 부분배열을 병합해 정렬된 배열 하나로 만든다) 세가지 포인트가 존재. 
  - 유사코드
  ~~~
  Merge(A,p,q,r)  : A는 배열, p,q,r은 인덱스로 p <= q < r을 만족한다.
      n₁ = q-p+1 
      n₂ = r-q
      배열 L[1 .. n₁+1]과 R[1 .. n₂+1]을 생성한다.
      
      for i = 1 to n₁ 
          L[i] = A[p+i-1]
      
      for j = 1 to n₂  
          R[j] = A[q+j]
      
      L[n₁+1] = ∞
      R[n₂+1] = ∞
      i = 1
      j = 1
      
      for k = p to r
          if L[i] <= R[j]
              A[k] = L[i]
              i = i+1
          else A[k] = R[j]
              j = j+1
  ~~~
  - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/%ED%95%A9%EB%B3%91%EC%A0%95%EB%A0%AC.gif" width="500" height="300" />  
  
  - 성능 분석하기
  ~~~                                                                                     
     Merge(A,p,r)
        if p < r
            q = ⌊(p+r)/2⌋
            Merge(A,p,q)
            Merge(A,q+1,r)
            Merge(A,p,q,r)
  ~~~
      > if n이 충분히 작아 상수 C보다 작을 때, T(n) = Θ(1)
      > 다른 모든 경우엔, T(n) = aT(n/b) + D(n) + C(n)
      *) 지금까지 a=b인 상황을 봤지만, 같지 않은 상황에서도 적용이 가능한 알고리즘도 분석할 예정  
      *) D(n)은 문제를 분할하는데 걸리는 시간, C(n) 부분 문제들의 해를 결합하여 원래 문제의 해를 만드는데 걸리는 시간  
      
      
      > 분할 : 
 <hr/>
