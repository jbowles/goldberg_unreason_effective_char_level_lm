using DataStructures
using BenchmarkTools
shakespeare_input() = "http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt"
linux_input() = "http://cs.stanford.edu/people/karpathy/char-rnn/linux_input.txt"
file_namer(url::String) = split(url,"/")[end]
file_down(url::String,file::String) = download(url,file)
padding(str::String,num::Int64) = repeat(str, num)
rewind(str::String,len::Int64) = str[end-(len-1):end]

#acummType = Accumulator{Char,Int}
modelType = Dict{String,Array{Tuple{Char,Float64},1}}

#sum over the accumulator values
#array tuple of char and count/total percentage
function normalize(accum)
   s = float(sum(accum))
   [(char,float(sum(count))/s) for (char,count) in accum]
end
#lm = train_char_lm(shakespeare_input())
function train_char_lm(url::String;order::Int64=4)::modelType
   file = file_namer(url)
   if !isfile(file)
      file_down(url, file)
   end
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

function generate_letter(model::modelType, history::String, order::Int64)::Char
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

function generate_text(model::modelType, order::Int64; nletters::Int64=1000)::String
   history = padding("~", order)
   out = String[]
   for i in 1:nletters
      c = string(generate_letter(model, history, order))
      history = rewind(history,order) * c
      push!(out,c)
   end
   join(out,"")
end


#@btime train_char_lm(shakespeare_input())
#lm = train_char_lm(shakespeare_input())
#println(generate_text(lm, 4))

#lm = train_char_lm(shakespeare_input(),order=2)
#txt = generate_text(lm, 2)




#BenchmarkTools.Trial:
#  memory estimate:  310.16 MiB
#  allocs estimate:  10931898
#  --------------
#  minimum time:     2.665 s (11.27% GC)
#  median time:      2.699 s (12.34% GC)
#  mean time:        2.699 s (12.34% GC)
#  maximum time:     2.733 s (13.39% GC)
#  --------------
#  samples:          2
#  evals/sample:     1

#BenchmarkTools.Trial:
#  memory estimate:  293.24 MiB
#  allocs estimate:  9822504
#  --------------
#  minimum time:     1.967 s (16.97% GC)
#  median time:      2.006 s (17.88% GC)
#  mean time:        2.005 s (17.80% GC)
#  maximum time:     2.040 s (18.53% GC)
#  --------------
#  samples:          3
#  evals/sample:     1
