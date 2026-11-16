#include <ql/quantlib.hpp>
#include <iostream>
#include <iomanip>
using namespace QuantLib;
int main(){
  Calendar cal=TARGET(); DayCounter dc=Actual365Fixed();
  Date today(3,June,2004); Settings::instance().evaluationDate()=today;
  Date settlement=cal.advance(today,3,Days);
  RelinkableHandle<YieldTermStructure> ts; ts.linkTo(ext::make_shared<FlatForward>(settlement,0.032,dc));
  auto model=ext::make_shared<HullWhite>(ts);
  Date issue=cal.adjust(today-100*Days); Date mat=cal.advance(issue,10,Years);
  Schedule s=MakeSchedule().from(issue).to(mat).withCalendar(cal).withFrequency(Semiannual).withConvention(ModifiedFollowing).withRule(DateGeneration::Backward);
  auto eng=ext::make_shared<TreeCallableFixedRateBondEngine>(model,240,ts);
  auto price=[&](const CallabilitySchedule& ex){CallableFixedRateBond b(3,10000.0,s,std::vector<Rate>(1,0.05),Thirty360(Thirty360::BondBasis),ModifiedFollowing,100.0,issue,ex); b.setPricingEngine(eng); return b.cleanPrice();};
  Date cd=s.date(8); std::cout<<std::setprecision(17)<<"couponDate "<<cd<<"\n";
  for(auto t:{Callability::Call,Callability::Put}){ Real k=t==Callability::Call?100.0:110.0, u=t==Callability::Call?200.0:0.0;
    CallabilitySchedule on; on.push_back(ext::make_shared<Callability>(Bond::Price(k,Bond::Price::Clean),t,cd));
    std::cout<<"expected "<<price(on)<<"\n";
    for(int d=1;d<=7;++d){CallabilitySchedule ex; ex.push_back(ext::make_shared<Callability>(Bond::Price(u,Bond::Price::Dirty),t,cd-d)); ex.push_back(on.front()); std::cout<<" d"<<d<<" "<<price(ex)<<"\n";}}
}
