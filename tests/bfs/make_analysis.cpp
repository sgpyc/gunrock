#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>

using namespace std;

typedef int SizeT;

int num_gpus = 0;
int num_iterations = 0;

template <typename Type>
bool IsPure2(Type x)
{
    while (x>2)
    {
        if ((x%2) != 0) return false;
        x /= 2;
    }
    return true;
}

template <typename Type>
void Delete_Space(Type* &space)
{
    if (space == NULL) return;
    delete[] space; space = NULL;
}

template <typename Type>
void Delete_Space(SizeT dim1, Type** &space)
{
    if (space == NULL) return;
    for (SizeT i=0; i<dim1; i++)
    {
        Delete_Space(space[i]);
    }
    delete[] space; space = NULL;
}

template <typename Type>
void Delete_Space(SizeT dim1, SizeT dim2, Type*** &space)
{
    if (space == NULL) return;
    for (SizeT i=0; i<dim1; i++)
        Delete_Space(dim2, space[i]);
    delete[] space; space = NULL;
}

template <typename Type>
void New_Space(SizeT dim1, Type* &space, Type val = -1)
{
    if (space != NULL) Delete_Space(space);
    space = new Type[dim1];
    for (SizeT i=0; i<dim1; i++)
        space[i] = val;
}

template <typename Type>
void New_Space(SizeT dim1, SizeT dim2, Type** &space, Type val = -1)
{
    if (space != NULL) Delete_Space(dim1, space);
    space = new Type*[dim1];
    for (SizeT i=0; i<dim1; i++)
    {
        space[i] = NULL;
        New_Space(dim2, space[i], val);
    }
}

template <typename Type>
void New_Space(SizeT dim1, SizeT dim2, SizeT dim3, Type*** &space, Type val = -1)
{
    if (space != NULL) Delete_Space(dim1, dim2, space);
   
    space = new Type**[dim1];
    for (SizeT i=0; i<dim1; i++)
    {
        space[i] = NULL;
        New_Space(dim2, dim3, space[i], val);
    }
}

SizeT GetLastNum(char str[])
{
    char num_str[32] = "";
    int  num_length = 0;
    int  i = strlen(str) - 1;
    bool in_num = false;
    char ch = '\0';
    SizeT x = 0;

    while (i>=0)
    {
        ch = str[i];
        if (ch >= '0' && ch <= '9')
        {
            in_num = true;
            num_str[num_length] = ch;
            num_length++;
        } else {
            if (in_num) break;
        }
        i--;
    }

    num_str[num_length] = '\0';
    for (i=0; i<num_length/2; i++)
    {
        ch = num_str[i];
        num_str[i] = num_str[num_length-1-i];
        num_str[num_length-1-i] = ch;
    }
    x = atoi(num_str);
    return x;
}

void Str2Length(char str[], SizeT ***space, int **counter)
{
    SizeT dim1, dim2;
    SizeT length = -1;

    sscanf(str, "%d %d", &dim2, &dim1);
    length = GetLastNum(str);

    if (counter[dim1][dim2] == 0)
    {
        space[dim1][dim2] = NULL;
        New_Space(1, space[dim1][dim2]);
    } else if (IsPure2(counter[dim1][dim2]))
    {
        SizeT* temp_space = NULL;
        New_Space(counter[dim1][dim2] * 2, temp_space);
        memcpy(temp_space, space[dim1][dim2], 
            sizeof(SizeT) * counter[dim1][dim2]);
        Delete_Space(space[dim1][dim2]);
        space[dim1][dim2] = temp_space;
    }
    //cout<<dim1<<","<<dim2<<","<<counter[dim1][dim2]<<" -> "<<length<<endl;
    space[dim1][dim2][counter[dim1][dim2]] = length;
    counter[dim1][dim2] ++;
}

void Str2Length(char str[], SizeT ***space)
{
    SizeT dim1, dim2, dim3;
    SizeT length = -1;

    sscanf(str, "%d %d %d", &dim2, &dim1, &dim3);
    length = GetLastNum(str);
    //cout<<dim1<<","<<dim2<<","<<dim3<<" -> "<<length<<endl;
    space[dim1][dim2][dim3] = length;
}

void Str2Length(char str[], SizeT **space)
{
    SizeT dim1, dim2;
    SizeT length = -1;

    sscanf(str, "%d %d", &dim2, &dim1);
    length = GetLastNum(str);
    //cout<<dim1<<","<<dim2<<" -> "<<length<<endl;
    space[dim1][dim2] = length;
}

void Length2Str(int counter, SizeT* length, char str[])
{
    strcpy(str, "");
    //printf("counter = %d\n", counter);fflush(stdout);
    for (int i=0; i<counter; i++)
    {
        if (length[i] < 0) continue;
        //printf("length[%d] = %d\n", i, length[i]);fflush(stdout);
        sprintf(str + strlen(str), "%s%d", i==0?"":",", length[i]);
    }
}

void Length2Str(int ite, int gpu, int** counter, SizeT*** length, char str[])
{
    Length2Str(counter[ite][gpu], length[ite][gpu], str);
}

int main(int argc, char* argv[])
{
    SizeT ***input_in__length  = NULL;
    int   ** input_in__counter = NULL;
    int   ** input_target      = NULL;
    SizeT ***input_out_length  = NULL;
    int   ** input_out_counter = NULL;
    SizeT ***subq__in__length  = NULL;
    int   ** subq__in__counter = NULL;
    int   ** subq__target      = NULL;
    SizeT ***subq__out_length  = NULL;
    int   ** subq__out_counter = NULL;
    SizeT ***fullq_in__length  = NULL;
    int   ** fullq_in__counter = NULL;
    int   ** fullq_target      = NULL;
    SizeT ***fullq_out_length  = NULL;
    int   ** fullq_out_counter = NULL;
    SizeT ***outpu_out_length  = NULL;

    num_gpus = atoi(argv[1]);
    num_iterations = atoi(argv[2]);
    New_Space<int   >(num_iterations, num_gpus, input_in__counter, 0   );
    New_Space<SizeT*>(num_iterations, num_gpus, input_in__length , NULL);
    New_Space<int   >(num_iterations, num_gpus, input_target     , -1  );
    New_Space<int   >(num_iterations, num_gpus, input_out_counter, 0   );
    New_Space<SizeT*>(num_iterations, num_gpus, input_out_length , NULL);
    New_Space<int   >(num_iterations, num_gpus, subq__in__counter, 0   );
    New_Space<SizeT*>(num_iterations, num_gpus, subq__in__length , NULL);
    New_Space<int   >(num_iterations, num_gpus, subq__target     , -1  );
    New_Space<int   >(num_iterations, num_gpus, subq__out_counter, 0   );
    New_Space<SizeT*>(num_iterations, num_gpus, subq__out_length , NULL);
    New_Space<int   >(num_iterations, num_gpus, fullq_in__counter, 0   );
    New_Space<SizeT*>(num_iterations, num_gpus, fullq_in__length , NULL);
    New_Space<int   >(num_iterations, num_gpus, fullq_target     , -1  );
    New_Space<int   >(num_iterations, num_gpus, fullq_out_counter, 0   );
    New_Space<SizeT*>(num_iterations, num_gpus, fullq_out_length , NULL);
    New_Space(num_iterations, num_gpus, num_gpus, outpu_out_length, -1);

    ifstream fin(argv[3]);
    char str[512], str2[512];
    while (!fin.eof())
    {
        fin.getline(str, 512);

        if (strstr(str, "CORR")) {
            cout<<str<<endl;
        } else if (strstr(str, "From"      )) {
            //cout<<str<<endl;
            Str2Length(str, input_in__length, input_in__counter);
        } else if (strstr(str, "GotInput")) {
            //cout<<str<<endl;
            Str2Length(str, input_out_length, input_out_counter);
        } else if (strstr(str, "stage1"  )) {
            //cout<<str<<endl;
            Str2Length(str, subq__in__length, subq__in__counter);
        } else if (strstr(str, "pushed to fullq")) {
            //cout<<str<<endl;
            Str2Length(str, subq__out_length, subq__out_counter);
        } else if (strstr(str, "Got job.")) {
            //cout<<str<<endl;
            Str2Length(str, fullq_in__length, fullq_in__counter);
        } else if (strstr(str, "Fullqueue finished")) {
            //cout<<str<<endl;
            Str2Length(str, fullq_out_length, fullq_out_counter);
        } else if (strstr(str, "out_length")) {
            //cout<<str<<endl;
            Str2Length(str, outpu_out_length);
        } else if (strstr(str, "target_input_count[0] ->") ||
                   strstr(str, "target_input_count[1] ->"))
        {
            if        (strstr(str, "i_queues"   ))
            {
                Str2Length(str, input_target);
            } else if (strstr(str, "subq__queue"))
            {
                Str2Length(str, subq__target);
            } else if (strstr(str, "fullq_queue"))
            {
                Str2Length(str, fullq_target);
            }
        }
    }
    fin.close();

    for (int i=0; i<num_iterations; i++)
    {
        bool to_show = false;
        for (int gpu=0; gpu<num_gpus; gpu++)
        {
            if (input_in__counter[i][gpu] != 0) to_show = true;
            if (input_target     [i][gpu] >= 0) to_show = true;
            if (input_out_counter[i][gpu] != 0) to_show = true;
            if (subq__in__counter[i][gpu] != 0) to_show = true;
            if (subq__target     [i][gpu] >= 0) to_show = true;
            if (subq__out_counter[i][gpu] != 0) to_show = true;
            if (fullq_in__counter[i][gpu] != 0) to_show = true;
            if (fullq_target     [i][gpu] >= 0) to_show = true;
            if (fullq_out_counter[i][gpu] != 0) to_show = true;
            for (int peer=0; peer<num_gpus; peer++)
                if (outpu_out_length[i][gpu][peer] >= 0) to_show = true;
            if (to_show) break;
        }
        if (to_show)
        {
            for (int gpu=0; gpu<num_gpus; gpu++)
            {
                cout<<i<<"\t"<<gpu<<"\t|";

                Length2Str(i, gpu, input_in__counter, input_in__length, str );
                Length2Str(i, gpu, input_out_counter, input_out_length, str2);
                cout<<str<<"\t"<<input_target[i][gpu]<<"\t"<<str2<<"\t|";

                Length2Str(i, gpu, subq__in__counter, subq__in__length, str );
                Length2Str(i, gpu, subq__out_counter, subq__out_length, str2);
                cout<<str<<"\t"<<subq__target[i][gpu]<<"\t"<<str2<<"\t|";

                Length2Str(i, gpu, fullq_in__counter, fullq_in__length, str );
                Length2Str(i, gpu, fullq_out_counter, fullq_out_length, str2);
                cout<<str<<"\t"<<fullq_target[i][gpu]<<"\t"<<str2<<"\t|";

                Length2Str(num_gpus, outpu_out_length[i][gpu], str);
                cout<<str<<endl;
            }
            cout<<"-------------------------"<<endl;
        }
    }

    Delete_Space(num_iterations, num_gpus, input_in__length); 
    Delete_Space(num_iterations, num_gpus, input_out_length); 
    Delete_Space(num_iterations, num_gpus, subq__in__length); 
    Delete_Space(num_iterations, num_gpus, subq__out_length); 
    Delete_Space(num_iterations, num_gpus, fullq_in__length); 
    Delete_Space(num_iterations, num_gpus, fullq_out_length); 
    Delete_Space(num_iterations, num_gpus, outpu_out_length); 
    return 0;
}

