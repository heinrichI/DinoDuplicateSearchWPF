using System.Windows;
using DinoDuplicateSearch.ViewModels;

namespace DinoDuplicateSearch.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        var vm = (MainViewModel)DataContext;
        vm.Search.SearchCompleted += groups =>
        {
            vm.Results.Groups = groups;
        };
        vm.Search.SwitchToResults += idx =>
        {
            vm.SelectedTabIndex = idx;
        };
    }
}
