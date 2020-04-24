from django.shortcuts import render
from django.utils import timezone
from .models import Post, Param
from django.shortcuts import render, get_object_or_404
from .forms import PostForm
from django.shortcuts import redirect
from .calc.predref_direct import lh_model
from .calc.predref_direct import power_model


def post_list(request):
    posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request, 'blog/post_list.html', {'posts': posts})

def problem_result(request, pk):
    post = get_object_or_404(Post, pk=pk)

    if post.problem_id == 1:
        Eref = request.POST['Eref']
        k_ref = request.POST['k_ref']
        Emet = request.POST['Emet']
        k_met = request.POST['k_met']
        degree = request.POST['degree']
        B = request.POST['B']
        lh_model.dir_problem(Eref, k_ref, Emet, k_met, degree, B)
        return render(request, 'blog/result_lang.html', {'post': post})

    if post.problem_id == 2:
        Eref = request.POST['Eref']
        k_ref = request.POST['k_ref']
        Emet = request.POST['Emet']
        k_met = request.POST['k_met']
        degree = request.POST['degree']
        power_model.dir_problem_power(Eref, k_ref, Emet, k_met, degree)
        return render(request, 'blog/result_power.html', {'post': post})


def post_detail(request, pk):
    post = get_object_or_404(Post, pk=pk)
    params = Param.objects.filter(problem=post)

    print(params)
    return render(request, 'blog/post_detail.html', {'post': post, 'params': params})

def post_edit(request, pk):
    post = get_object_or_404(Post, pk=pk)
    if request.method == "POST":
        form = PostForm(request.POST, instance=post)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.published_date = timezone.now()
            post.save()
            return redirect('post_detail', pk=post.pk)
    else:
        form = PostForm(instance=post)
    return render(request, 'blog/post_edit.html', {'form': form})

def post_new(request):
    if request.method == "POST":
        form = PostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.published_date = timezone.now()
            post.save()
            return redirect('post_detail', pk=post.pk)
    else:
        form = PostForm()
    return render(request, 'blog/post_edit.html', {'form': form})