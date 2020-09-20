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
<hr/>

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
   for j = 2 to A.length                  c₁            bn
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
  - 유사코드 (보조 프로시저인 "결합" 과정)
  ~~~
  Merge(A,p,q,r)  : A는 배열, p,q,r은 인덱스로 p <= q < r을 만족한다.                  cost           times
      n₁ = q-p+1                                                                      c₁             1
      n₂ = r-q                                                                        c₂             1
      배열 L[1 .. n₁+1]과 R[1 .. n₂+1]을 생성한다                                      c₃             1
      
      for i = 1 to n₁                                                                 c₄             n₁
          L[i] = A[p+i-1]                                                             c₅             1
      
      for j = 1 to n₂                                                                 c₆             n₂
          R[j] = A[q+j]                                                               c₇             1
      
      L[n₁+1] = ∞                                                                     c₈             1
      R[n₂+1] = ∞                                                                     c₉             1
      i = 1                                                                           c₁₀            1
      j = 1                                                   
      
      for k = p to r                                                                  c₁₁            n
          if L[i] <= R[j]                                                             c₁₂            c
              A[k] = L[i]                                                             c₁₃            1
              i = i+1                                                                 c₁₄            1
          else A[k] = R[j]                                                            c₁₅            1
              j = j+1                                                                 c₁₆            1
  ~~~
      > T(n) = c₄n₁ + c₆n₂ + c₁₁n + C
      > T(n) = an + b = Θ(n)
     
  
  - 유사코드 (합병정렬 전체 과정)
  ~~~                                                                                     
     Merge-Sort(A,p,r)
        if p < r
            q = ⌊(p+r)/2⌋
            Merge-Sort(A,p,q)
            Merge-Sort(A,q+1,r)
            Merge(A,p,q,r)
  ~~~
      # 분할정복 분석
      > 크기 n이 충분히 작아 n <= c 조건을 만족할 경우, T(n) = Θ(1)
      > 다른 모든 경우엔, T(n) = aT(n/b) + D(n) + C(n)
      *) 지금까지 a=b인 상황을 봤지만, 같지 않은 상황에서도 적용이 가능한 알고리즘도 분석할 예정  
      *) D(n)은 문제를 분할하는데 걸리는 시간, C(n) 부분 문제들의 해를 결합하여 원래 문제의 해를 만드는데 걸리는 시간  
      
      # 합병정렬 분석
      > 분할 : 부분 배열의 중간 위치를 계산하는 과정, 상수 시간이 걸린다. (D(n) = Θ(1))
      > 정복 : 두 개의 부분 문제를 재귀적으로 풀면서, 각 부분 문제는 크기가 n/2로 줄어든다. (2T(n/2))
      > 결합 : 앞서 보았듯, Merge 프로시저는 Θ(n) 시간이 걸린다. (C(n) = Θ(n))
      따라서, T(n) = 2T(n/2) + Θ(n) (n>1) / T(n) = Θ(1) (n=1)
      
      
  - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/%ED%95%A9%EB%B3%91%EC%A0%95%EB%A0%AC.gif" width="500" height="300" />  
 <hr/>
 
### 알고리즘의 효율성
1. 점근적 분석
  > 알고리즘의 효율성을 판단할때, 주로 수행시간이 얼마나 걸리는지를 중심으로 판단한다.  
    이때, 각 알고리즘별 정확한 수행시간을 중심으로 판단하는 것이 아닌, 입력의 크기가 극한으로 증가할 때 수행시간을 중심으로 판단한다.
  > 이는 입력의 크기가 아주 작은 경우를 제외하고, 대부분의 경우에 좋은 판단기법이 되며 이러한 방식을 점근적 분석이라 한다.  
  > 이때, 입력의 크기가 극한으로 커가므로, 수행시간의 증가 차수 중 상수계수나 저차항은 무시된다.

2. 점근적 표기
  > 위에서 살펴본 점근적 분석을 형식적으로 표기하는 방법을 의미한다.
<hr/>

### 점근적 표기
- 알고리즘의 점근적 수행시간을 나타내는데 사용하는 표기.
- 정의역의 경우 자연수의 집합 N ={0,1,2 ...}로 정의된다.
  
  1. O 표기 (빅오 표기)  
    > 명칭 : 주어진 함수 g(n)에 대한 빅오g(n) 또는 오g(n)이라 명명한다. (O(g(n)))  
    > 정의 : O(g(n)) = { f(n) : 모든 n > n₀에 대해 0 <= f(n) <= cg(n)인 양의 상수 c, n₀이 존재한다. }  
           : f(n) = O(g(n)): g(n)는 f(n)에 대한 점근 상한을 의미한다.  
    > 함수의 상한을 나타내기 위해 사용되는 개념이다.  
      O는 알고리즘의 최악의 실행 시간을 설명하는데 좋다.  
    +) 추가이해  
    https://ko.wikipedia.org/wiki/%EC%A0%90%EA%B7%BC_%ED%91%9C%EA%B8%B0%EB%B2%95  
    
  2. Ω표기 (오메가 표기)  
    > 명칭 : 주어진 함수 g(n)에 대한 빅 오메가 g(n) 또는 오메가 g(n)이라 명명한다. (Ω(g(n))  
    > 정의 : Ω(g(n)) = { f(n) : 모든 n > n₀에 대해 0 <= cg(n) <= f(n) 인 양의 상수 c, n₀이 존재한다.}  
           : f(n) = O(g(n)): g(n)는 f(n)에 대한 점근 하한을 의미한다.  
    > 함수의 하한을 나타내기 위해 사용되는 개념이다.  
      Ω은 알고리즘의 최상의 사례 실행 시간을 설명하는데 좋다.  
      
  3. Θ표기 (세타 표기)  
    > 명칭 : 주어진 함수 g(n)에 대해 세타g(n)이라 명명한다. (Θ(g(n))  
    > 정의 : Θ(g(n)) = = { f(n) : 모든 n > n₀에 대해 0 <= c₁g(n) <= f(n) <= c₂g(n)인 양의 상수 c₁, c₂, n₀이 존재한다.}  
           : 
           
<hr/>
### T(n) 증명하기  
  - 수행시간에 대한 점화식은 다양한 형태로 표기된다. 이렇게 "추측"된 점화식을 어떻게 증명할 수 있을까?
  - "치환법(substitution method)", "재귀 트리 방법(recursion-tree method", "마스터 방법(master method)"등의 방법을 통해 증명한다.
  - 보통 등식의 꼴을 띄지만, 간혹 부등식 형태의 점화식도 사용된다.(부등식의 경우 빅오 또는 오메가 표기를 통해 사용한다.)

* 치환법
  -수학적 귀납법을 사용한다.
    1) 초항에 대한 증명을 보여준다.
    2) n = k-1까지의 모든 경우에서 해당 점화식이 참임을 가정한다.
    3) n = k일때 점화식이 참임을 증명한다.
    ex)  
      ~~~
      T(n) = O(n*log n)을 증명하라
      1) 초항을 증명하라 / 정의역 n은 1보다 큰 자연수다. & T(1) = 1 & T(n) = 2T(n/2) + n
        n=2 > T(2)        <= 2*c*log₂2
              = 2T(1)+1   <= 2c
              = 4         <= 2c
              ∴ c >= 2이면 항상 참
      2) n = k-1까지 참이라고 가정
      3) n = k일때 참임을 증명하기
        T(k) = 2T(k/2) + k    > T(k/2)는 k-1까지 범위에 들어감. 즉 증명되어 있음.
                                ∴ T(k/2) <= c*(k/2)*log(k/2)
        2T(k/2) + k <= 2*(c*(k/2)*log(k/2)) + k
        T(k) <= c*k*logk - ck + k
        T(k) <= c*k*logk + (1-c)k
          > 1-c 는 항상 0보다 작은 수가 나온다. 
        ∴ T(k) <= c*k*logk + (1-c)k <= c*k*logk
           즉, T(k) <= c*k*logk (O(nlogn))을 만족한다.
      ~~~

* 재귀 트리 방법
 - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/%ED%95%A9%EB%B3%91%EC%A0%95%EB%A0%AC_%EC%9E%AC%EA%B7%80%ED%8A%B8%EB%A6%AC.png" width="500" height="300" />
 - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/%ED%95%A9%EB%B3%91%EC%A0%95%EB%A0%AC_%EC%9E%AC%EA%B7%80%ED%8A%B8%EB%A6%AC2.png" width="500" height="300" />
 - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/%ED%95%A9%EB%B3%91%EC%A0%95%EB%A0%AC_%EC%9E%AC%EA%B7%80%ED%8A%B8%EB%A6%AC3.png" width="500" height="300" /> 
  ex)
   ~~~
   
   ~~~
  

      
