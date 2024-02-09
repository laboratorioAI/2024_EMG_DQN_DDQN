function confusion2latex(outputFile, mat, totalCols, porcentCols...
    , totalRows, porcentRows, total, percentTotal)
%confusion2latex 
%
% Inputs
%   nameUser        char con el nombre de la carpeta del usuario.
%
% Outputs
%   nameUser        char con el nombre de la carpeta del usuario.
% 
% Ejemplo
%    = confusion2latex()
%

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: z_tja
jonathan.a.zea@ieee.org
Cuando escribí este código,  solo dios y yo sabíamos como funcionaba.
Ahora solo lo sabe dios.

"When a measure becomes a target,  it ceases to be a good measure"
-Charles Goodhart.

04 June 2020
Matlab 9.7.0.1261785 (R2019b) Update 3.
%}

%%
cols = [totalCols; porcentCols];
cols = cols(:);

%%
fid = fopen(outputFile, 'w');

fprintf(fid, '%% add this packages at the top of the latex doc.\n');
fprintf(fid, '%%\\usepackage{adjustbox}\n');
fprintf(fid, '%%\\usepackage{graphicx}\n');
fprintf(fid, '%%\\usepackage{multirow}\n');
fprintf(fid, '%%\\usepackage[table,xcdraw]{xcolor}\n');
fprintf(fid, '\n');
fprintf(fid, '\n');
fprintf(fid, '\\begin{table*}[htbp] %% recommended when the paper has 2 columns.\n');
fprintf(fid, '%% \\begin{table}[htbp] %% (Recommended to uncomment when the paper has 1 column.)\n');
fprintf(fid, '\n');
fprintf(fid, '\\caption{Confusion Matrix of the Proposed Models} \\label{tab:confMat}\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '%%\\begin{adjustbox}{scale = 0.625} %% caso IEEE. Solo en ambiente table! Se puede añadir (",center")\n');
fprintf(fid, '%%\\small\n');
fprintf(fid, '\\begin{tabular}{c|cccccc|c}\n');
fprintf(fid, '\\hline\n');
fprintf(fid, '                                   \t& \\multicolumn{6}{c|}{}                             \t&               \\\\\n');
fprintf(fid, '\\multirow{-2}{*}{}                \t& \\multicolumn{6}{c|}{\\multirow{-2}{*}{Targets}}   \t&               \\\\\\cline{2-7}\n');
fprintf(fid, '                                   \t& waveIn                                             \t& waveOut       \t& fist         \t& open           \t& pinch         \t& noGesture      \t& \\multirow{-3}{*}{\\textbf{\\begin{tabular}[c]{@{}c@{}}Predictions Count\\\\ (Precision)\\end{tabular}}}   \\\\ \\hline\n');
fprintf(fid, '\\multicolumn{1}{c|}{waveIn}       \t& \\textbf{%d}                                       \t& %d            \t& %d           \t& %d             \t& %d            \t& %d             \t& \\textbf{\\begin{tabular}[c]{@{}c@{}}%d\\\\ %.4g\\%%\\end{tabular}}                                                          \\\\\\hline\n', mat(1, :), totalRows(1), porcentRows(1));
fprintf(fid, '\\multicolumn{1}{c|}{waveOut}      \t& %d                                                 \t& \\textbf{%d}  \t& %d           \t& %d             \t& %d            \t& %d             \t& \\textbf{\\begin{tabular}[c]{@{}c@{}}%d\\\\ %.4g\\%%\\end{tabular}}                                                          \\\\\\hline\n', mat(2, :), totalRows(2), porcentRows(2));
fprintf(fid, '\\multicolumn{1}{c|}{fist}         \t& %d                                                 \t& %d            \t& \\textbf{%d} \t& %d             \t& %d            \t& %d             \t& \\textbf{\\begin{tabular}[c]{@{}c@{}}%d\\\\ %.4g\\%%\\\\\\end{tabular}}                                                      \\\\\\hline\n', mat(3, :), totalRows(3), porcentRows(3));
fprintf(fid, '\\multicolumn{1}{c|}{open}         \t& %d                                                 \t& %d            \t& %d           \t& \\textbf{%d}   \t& %d            \t& %d             \t& \\textbf{\\begin{tabular}[c]{@{}c@{}}%d\\\\ %.4g\\%%\\\\\\end{tabular}}                                                      \\\\\\hline\n', mat(4, :), totalRows(4), porcentRows(4));
fprintf(fid, '\\multicolumn{1}{c|}{pinch}        \t& %d                                                 \t& %d            \t& %d           \t& %d             \t& \\textbf{%d}  \t& %d             \t& \\textbf{\\begin{tabular}[c]{@{}c@{}}%d\\\\ %.4g\\%% \\\\\\end{tabular}}                                                    \\\\ \\hline\n', mat(5, :), totalRows(5), porcentRows(5));
fprintf(fid, '\\multicolumn{1}{c|}{noGesture}    \t& %d                                                 \t& %d            \t& %d           \t& %d             \t& %d            \t& \\textbf{%d}   \t& \\textbf{\\begin{tabular}[c]{@{}c@{}}%d\\\\ %.4g\\%%\\end{tabular}}                                                          \\\\ \\hline\\hline\n', mat(6, :), totalRows(6), porcentRows(6));
fprintf(fid, '\\textbf{\\begin{tabular}[c]{@{}c@{}}Targets Count\\\\ (Sensitivity)\\end{tabular}}                        \t& \\textbf{\\begin{tabular}[c]{@{}c@{}}%d\\\\ %.4g\\%%\\end{tabular}}              \t& \\textbf{\\begin{tabular}[c]{@{}c@{}}%d\\\\ %.4g\\%%\\end{tabular}}             \t& \\textbf{\\begin{tabular}[c]{@{}c@{}}%d\\\\ %.4g\\%%\\end{tabular}}             \t& \\textbf{\\begin{tabular}[c]{@{}c@{}}%d\\\\ %.4g\\%%\\end{tabular}}         \t& \\textbf{\\begin{tabular}[c]{@{}c@{}}%d\\\\ %.4g\\%%\\end{tabular}}         \t& \\textbf{\\begin{tabular}[c]{@{}c@{}}%d\\\\ %.4g\\%%\\end{tabular}}         \t& {\\color[HTML]{CB0000} \\textbf{\\begin{tabular}[c]{@{}c@{}}%d\\\\ %.4g\\%%\\end{tabular}}}                \\\\ \\hline\n', ...
    cols , total, percentTotal);
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '%%\\end{adjustbox}\n');
fprintf(fid, '%%\\end{table}\n');
fprintf(fid, '\\end{table*}\n');

fclose(fid);
