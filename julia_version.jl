using DataStructures
using BenchmarkTools
shakespeare_input() = "http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt"
linux_input() = "http://cs.stanford.edu/people/karpathy/char-rnn/linux_input.txt"
file_namer(url) = split(url,"/")[end]
file_down(url,file) = download(url,file)
padding(chr,num) = repeat(chr, num)
rewind(str,len) = str[end-(len-1):end]

#sum over the accumulator values
#array tuple of char and count/total percentage
function normalize(accum)
   s = float(sum(accum))
   [(char,float(sum(count))/s) for (char,count) in accum]
end

function train_char_lm(url;order=4)
   file = file_namer(url)
   if !isfile(file)
      file_down(url, file)
   end
   #data = read(file, String)
   data = read(file, String)
   pad = padding("~",order)
   data = pad * data
   tlp = DefaultDict{String,Accumulator{Char,Int64}}(counter(String))
   for i in 1:(length(data)-order)
      hist,curr = data[i:(i-1)+order],data[i+order]
      tlp[hist][curr]+=1
   end
   Dict(hist => normalize(chars) for (hist,chars) in tlp)
end

function train_char_lm_inline(url;order=4)
   file = file_namer(url)
   if !isfile(file)
      file_down(url, file)
   end
   #data = read(file, String)
   data = read(file, String)
   pad = padding("~",order)
   data = pad * data
   tlp = DefaultDict{String,Accumulator{Char,Int64}}(counter(String))
   @inbounds for i in 1:(length(data)-order)
      hist,curr = data[i:(i-1)+order],data[i+order]
      tlp[hist][curr]+=1
   end
   Dict(hist => normalize(chars) for (hist,chars) in tlp)
end

function train_char_on_string(str;order=4)
   #data = read(file, String)
   data = str
   pad = padding("~",order)
   data = pad * data
   tlp = DefaultDict{String,Accumulator{Char,Int64}}(counter(String))
   for i in 1:(length(data)-order)
      hist,curr = data[i:(i-1)+order],data[i+order]
      tlp[hist][curr]+=1
   end
   Dict(hist => normalize(chars) for (hist,chars) in tlp)
end

function generate_letter(model, history, order)::Char
   history = rewind(history,order)
   dist = model[history]
   x = rand()
   for (c,v) in dist
      x=x-v
      if x <= 0.0
         return c
      end
   end
end

function generate_text(model, order; nletters=1000)
   history = padding("~", order)
   out = String[]
   for i in 1:nletters
      c = string(generate_letter(model, history, order))
      history = rewind(history,order) * c
      push!(out,c)
   end
   join(out,"")
end

#julia> @benchmark train_char_lm(shakespeare_input())
#BenchmarkTools.Trial:
#  memory estimate:  293.24 MiB
#  allocs estimate:  9822504
#  --------------
#  minimum time:     2.051 s (11.97% GC)
#  median time:      2.205 s (18.97% GC)
#  mean time:        2.175 s (17.15% GC)
#  maximum time:     2.268 s (20.07% GC)
#  --------------
#  samples:          3
#  evals/sample:     1
#
#julia> @benchmark train_char_lm_inline(shakespeare_input())
#BenchmarkTools.Trial:
#  memory estimate:  293.24 MiB
#  allocs estimate:  9822504
#  --------------
#  minimum time:     1.593 s (20.38% GC)
#  median time:      1.706 s (19.04% GC)
#  mean time:        1.821 s (18.78% GC)
#  maximum time:     2.164 s (17.78% GC)
#  --------------
#  samples:          3
#  evals/sample:     1
#
#julia> @benchmark train_char_lm(shakespeare_input())
#BenchmarkTools.Trial:
#  memory estimate:  293.24 MiB
#  allocs estimate:  9822504
#  --------------
#  minimum time:     1.739 s (18.91% GC)
#  median time:      1.837 s (18.90% GC)
#  mean time:        1.823 s (19.11% GC)
#  maximum time:     1.893 s (18.34% GC)
#  --------------
#  samples:          3
#  evals/sample:     1
#
#julia> @benchmark train_char_lm_inline(shakespeare_input())
#BenchmarkTools.Trial:
#  memory estimate:  293.24 MiB
#  allocs estimate:  9822504
#  --------------
#  minimum time:     1.947 s (19.55% GC)
#  median time:      2.218 s (17.62% GC)
#  mean time:        2.142 s (18.61% GC)
#  maximum time:     2.261 s (18.77% GC)
#  --------------
#  samples:          3
#  evals/sample:     1
#
#julia> @benchmark train_char_lm_inline(shakespeare_input())
#BenchmarkTools.Trial:
#  memory estimate:  293.24 MiB
#  allocs estimate:  9822504
#  --------------
#  minimum time:     2.196 s (17.52% GC)
#  median time:      2.287 s (18.04% GC)
#  mean time:        2.268 s (18.03% GC)
#  maximum time:     2.322 s (17.77% GC)
#  --------------
#  samples:          3
#  evals/sample:     1
#
#julia> @benchmark train_char_lm(shakespeare_input())
#BenchmarkTools.Trial:
#  memory estimate:  293.24 MiB
#  allocs estimate:  9822504
#  --------------
#  minimum time:     2.304 s (16.78% GC)
#  median time:      2.534 s (15.73% GC)
#  mean time:        2.463 s (16.97% GC)
#  maximum time:     2.550 s (18.37% GC)
#  --------------
#  samples:          3
#  evals/sample:     1
#
#julia> @benchmark train_char_lm(shakespeare_input(),order=7)
#BenchmarkTools.Trial:
#  memory estimate:  1.14 GiB
#  allocs estimate:  18693364
#  --------------
#  minimum time:     11.910 s (7.63% GC)
#  median time:      11.910 s (7.63% GC)
#  mean time:        11.910 s (7.63% GC)
#  maximum time:     11.910 s (7.63% GC)
#  --------------
#  samples:          1
#  evals/sample:     1
#
#julia> @benchmark train_char_lm_inline(shakespeare_input(),order=7)
#BenchmarkTools.Trial:
#  memory estimate:  1.14 GiB
#  allocs estimate:  18693364
#  --------------
#  minimum time:     11.942 s (9.51% GC)
#  median time:      11.942 s (9.51% GC)
#  mean time:        11.942 s (9.51% GC)
#  maximum time:     11.942 s (9.51% GC)
#  --------------
#  samples:          1
#  evals/sample:     1
