import Logo from './Logo';

const Header = () => {
  return (
    <header className="w-full py-6 px-4 sm:px-6 glass-effect">
      <div className="container flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Logo />
          <h1 className="text-2xl font-bold tracking-tight bg-gradient-to-r from-white to-white/70 bg-clip-text text-transparent">
            REF_INSIGHT
          </h1>
        </div>
      </div>
    </header>
  );
};

export default Header;
